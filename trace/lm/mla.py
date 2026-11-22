# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Multi-Head Latent Attention (MLA) canónica 100% Ternaria W1.58
Incorpora compresión latente KV, único kv_norm, RoPE desacoplado
y absorción completa de pesos (W_uk y W_uv) con soporte de caché INT8.
"""

import math
from trace.layers.bitlinear import BitLinear
from trace.layers.norm import RMSNorm
from trace.lm.config import TRACELMConfig

import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_rope_frequencies(dim: int, max_seq_len: int, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Precomputa cos y sin para codificación posicional rotacional desacoplada."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    cos = torch.cos(angles).repeat_interleave(2, dim=-1)
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    """
    Aplica rotación 2D sobre la dimensión de características.
    Convención intercalada (interleaved):
      pares en índices 0::2 y e impares en 1::2:
      x1 = x[..., 0::2], x2 = x[..., 1::2]
      rotated = [-x2, x1]
      output = x * cos + rotated * sin
    Esta convención se replica de forma idéntica en el motor nativo en C.
    """
    seq_len = x.shape[-2]
    assert start_pos + seq_len <= cos.shape[0], (
        f"Exceso de longitud RoPE: start_pos ({start_pos}) + seq_len ({seq_len}) > max ({cos.shape[0]})"
    )

    c = cos[start_pos : start_pos + seq_len, :].to(x.device)
    s = sin[start_pos : start_pos + seq_len, :].to(x.device)

    # x: [..., seq_len, dim]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rotated = torch.empty_like(x)
    rotated[..., 0::2] = -x2
    rotated[..., 1::2] = x1
    return x * c + rotated * s


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, cfg: TRACELMConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_nope_head_dim = cfg.qk_nope_head_dim
        self.qk_rope_head_dim = cfg.qk_rope_head_dim
        self.v_head_dim = cfg.v_head_dim
        self.scale = 1.0 / math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim)

        # 1. Proyecciones ternarias BitLinear (W1.58)
        self.w_dkv = BitLinear(cfg.d_model, cfg.kv_lora_rank, eps=cfg.eps, use_sub_ln=True)
        self.kv_norm = RMSNorm(cfg.kv_lora_rank, eps=cfg.eps)

        # w_uk y w_uv operan sobre c_kv ya normalizado por kv_norm (use_sub_ln=False, quantize_activations=False)
        self.w_uk = BitLinear(
            cfg.kv_lora_rank, cfg.n_heads * cfg.qk_nope_head_dim,
            eps=cfg.eps, use_sub_ln=False, quantize_activations=False
        )
        self.w_uv = BitLinear(
            cfg.kv_lora_rank, cfg.n_heads * cfg.v_head_dim,
            eps=cfg.eps, use_sub_ln=False, quantize_activations=False
        )
        self.w_kr = BitLinear(cfg.d_model, cfg.qk_rope_head_dim, eps=cfg.eps, use_sub_ln=True)

        self.w_q = BitLinear(cfg.d_model, cfg.n_heads * cfg.qk_nope_head_dim, eps=cfg.eps, use_sub_ln=True)
        self.w_qr = BitLinear(cfg.d_model, cfg.n_heads * cfg.qk_rope_head_dim, eps=cfg.eps, use_sub_ln=True)
        self.w_out = BitLinear(cfg.n_heads * cfg.v_head_dim, cfg.d_model, eps=cfg.eps, use_sub_ln=True)

        # Tablas RoPE precomputadas
        cos, sin = precompute_rope_frequencies(cfg.qk_rope_head_dim, cfg.max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_weight_absorption: bool = False,
        quantize_kv_int8: bool = False,
        add_sub_mode: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, S, _ = x.shape
        past_len = kv_cache[0].shape[1] if kv_cache is not None else 0

        # A. KV Compression y normalización con kv_norm ÚNICO
        c_kv_new = self.kv_norm(self.w_dkv(x, add_sub_mode=add_sub_mode)) # [B, S, kv_lora_rank]
        if quantize_kv_int8:
            scale_kv = c_kv_new.abs().amax(dim=-1, keepdim=True).clamp(min=self.cfg.eps) / 127.0
            c_kv_new = torch.clamp(torch.round(c_kv_new / scale_kv), -127.0, 127.0) * scale_kv

        # Decoupled RoPE Key: apply_rope con start_pos = past_len
        k_pe_raw = self.w_kr(x, add_sub_mode=add_sub_mode) # [B, S, qk_rope_head_dim]
        k_pe_new = apply_rope(k_pe_raw, self.rope_cos, self.rope_sin, start_pos=past_len)

        if kv_cache is not None:
            c_kv_past, k_pe_past = kv_cache
            c_kv = torch.cat([c_kv_past, c_kv_new], dim=1)
            k_pe = torch.cat([k_pe_past, k_pe_new], dim=1)
        else:
            c_kv = c_kv_new
            k_pe = k_pe_new
        new_kv_cache = (c_kv.detach(), k_pe.detach())

        total_S = c_kv.shape[1]

        # B. Consultas: Contenido y RoPE (transposición ANTES de apply_rope para que dim=-2 sea S)
        q_c = self.w_q(x, add_sub_mode=add_sub_mode).view(B, S, self.n_heads, self.qk_nope_head_dim).transpose(1, 2)
        q_pe_raw = self.w_qr(x, add_sub_mode=add_sub_mode).view(B, S, self.n_heads, self.qk_rope_head_dim).transpose(1, 2)
        q_pe = apply_rope(q_pe_raw, self.rope_cos, self.rope_sin, start_pos=past_len)

        # C. Obtener pesos efectivos ternarios usando get_effective_weight()
        w_uk_eff = self.w_uk.get_effective_weight()
        w_uv_eff = self.w_uv.get_effective_weight()

        w_uk_heads = w_uk_eff.view(self.n_heads, self.qk_nope_head_dim, self.kv_lora_rank)
        w_uv_heads = w_uv_eff.view(self.n_heads, self.v_head_dim, self.kv_lora_rank)

        # D. Cálculo de Atención
        if use_weight_absorption:
            # 1. Absorbe w_uk en q_c
            q_c_absorbed = torch.einsum("b h s d, h d c -> b h s c", q_c, w_uk_heads)
            scores_c = torch.matmul(q_c_absorbed, c_kv.unsqueeze(1).transpose(-1, -2))
        else:
            k_c = F.linear(c_kv, w_uk_eff).view(B, total_S, self.n_heads, self.qk_nope_head_dim).transpose(1, 2)
            scores_c = torch.matmul(q_c, k_c.transpose(-1, -2))

        scores_pe = torch.matmul(q_pe, k_pe.unsqueeze(1).transpose(-1, -2))
        attn_scores = (scores_c + scores_pe) * self.scale

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask[:, :, :, :total_S]

        attn_weights = F.softmax(attn_scores, dim=-1)

        # E. Agregación de valores: Absorción completa de w_uv
        if use_weight_absorption:
            # o_latent en espacio latente [B, H, S, kv_lora_rank]
            o_latent = torch.matmul(attn_weights, c_kv.unsqueeze(1))
            # Proyecta únicamente los S tokens nuevos evitando descomprimir total_S
            out = torch.einsum("b h s c, h v c -> b h s v", o_latent, w_uv_heads)
        else:
            v = F.linear(c_kv, w_uv_eff).view(B, total_S, self.n_heads, self.v_head_dim).transpose(1, 2)
            out = torch.matmul(attn_weights, v) # [B, H, S, v_head_dim]

        out = out.transpose(1, 2).contiguous().view(B, S, self.n_heads * self.v_head_dim)
        return self.w_out(out, add_sub_mode=add_sub_mode), new_kv_cache
