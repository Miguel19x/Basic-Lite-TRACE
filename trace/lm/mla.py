# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available

"""
Multi-Head Latent Attention (MLA) canónica 100% Ternaria W1.58.
Incorpora compresión latente KV, único kv_norm, RoPE desacoplado intercalado (0::2 / 1::2),
SDPA nativo para entrenamiento/prefill en GPU y absorción completa de pesos (W_uk y W_uv)
con soporte de caché INT8 simétrico [-127, 127].
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from trace.layers.bitlinear import BitLinear
from trace.layers.norm import RMSNorm
from trace.lm.config import TRACELMConfig


def precompute_rope_frequencies(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precomputa cos y sin para RoPE intercalado (pares 0::2 y 1::2)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    cos = torch.cos(angles).repeat_interleave(2, dim=-1)
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)
    return cos, sin


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    start_pos: int = 0,
) -> torch.Tensor:
    """Aplica rotación 2D intercalada (0::2 / 1::2) en FP32 preservando x.dtype para AMP/SDPA."""
    seq_len = x.shape[-2]
    assert start_pos + seq_len <= cos.shape[0], (
        f"Posición RoPE fuera de rango: start_pos({start_pos}) + seq_len({seq_len}) > max_seq_len({cos.shape[0]})"
    )
    orig_dtype = x.dtype
    x_f32 = x.float()
    c = cos[start_pos : start_pos + seq_len, :].to(device=x.device)
    s = sin[start_pos : start_pos + seq_len, :].to(device=x.device)

    x1 = x_f32[..., 0::2]
    x2 = x_f32[..., 1::2]
    rotated = torch.empty_like(x_f32)
    rotated[..., 0::2] = -x2
    rotated[..., 1::2] = x1
    return (x_f32 * c + rotated * s).to(orig_dtype)


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

        q_en = cfg.quantize_weights_enabled
        w_sc = cfg.weight_scale_type

        self.w_dkv = BitLinear(
            cfg.d_model, cfg.kv_lora_rank, eps=cfg.eps, use_sub_ln=True,
            quantize_weights_enabled=q_en, weight_scale_type=w_sc,
        )
        self.kv_norm = RMSNorm(cfg.kv_lora_rank, eps=cfg.eps)

        self.w_uk = BitLinear(
            cfg.kv_lora_rank, cfg.n_heads * cfg.qk_nope_head_dim, eps=cfg.eps,
            use_sub_ln=False, quantize_activations=False,
            quantize_weights_enabled=q_en, weight_scale_type=w_sc,
        )
        self.w_uv = BitLinear(
            cfg.kv_lora_rank, cfg.n_heads * cfg.v_head_dim, eps=cfg.eps,
            use_sub_ln=False, quantize_activations=False,
            quantize_weights_enabled=q_en, weight_scale_type=w_sc,
        )
        self.w_kr = BitLinear(
            cfg.d_model, cfg.qk_rope_head_dim, eps=cfg.eps, use_sub_ln=True,
            quantize_weights_enabled=q_en, weight_scale_type=w_sc,
        )
        self.w_q = BitLinear(
            cfg.d_model, cfg.n_heads * cfg.qk_nope_head_dim, eps=cfg.eps, use_sub_ln=True,
            quantize_weights_enabled=q_en, weight_scale_type=w_sc,
        )
        self.w_qr = BitLinear(
            cfg.d_model, cfg.n_heads * cfg.qk_rope_head_dim, eps=cfg.eps, use_sub_ln=True,
            quantize_weights_enabled=q_en, weight_scale_type=w_sc,
        )
        self.w_out = BitLinear(
            cfg.n_heads * cfg.v_head_dim, cfg.d_model, eps=cfg.eps, use_sub_ln=True,
            quantize_weights_enabled=q_en, weight_scale_type=w_sc,
        )

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

        # A. Compresión KV y kv_norm único
        c_kv_new = self.kv_norm(self.w_dkv(x, add_sub_mode=add_sub_mode))
        if quantize_kv_int8:
            c_kv_f32 = c_kv_new.float()
            scale_kv = c_kv_f32.abs().amax(dim=-1, keepdim=True).clamp(min=self.cfg.eps) / 127.0
            c_kv_new = (torch.clamp(torch.round(c_kv_f32 / scale_kv), -127, 127) * scale_kv).to(c_kv_new.dtype)

        k_pe_raw = self.w_kr(x, add_sub_mode=add_sub_mode)
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

        # B. Consultas (transponer ANTES de RoPE para rotar sobre dim=-2 = S)
        q_c = self.w_q(x, add_sub_mode=add_sub_mode).view(B, S, self.n_heads, self.qk_nope_head_dim).transpose(1, 2)
        q_pe_raw = (
            self.w_qr(x, add_sub_mode=add_sub_mode)
            .view(B, S, self.n_heads, self.qk_rope_head_dim)
            .transpose(1, 2)
        )
        q_pe = apply_rope(q_pe_raw, self.rope_cos, self.rope_sin, start_pos=past_len)

        # C. Pesos efectivos W_uk y W_uv
        w_uk_eff = self.w_uk.get_effective_weight()
        w_uv_eff = self.w_uv.get_effective_weight()

        def _apply_mask(scores: torch.Tensor) -> torch.Tensor:
            if attn_mask is None:
                if S > 1:
                    causal_mask = torch.zeros((S, total_S), device=scores.device, dtype=scores.dtype)
                    causal_block = torch.triu(
                        torch.full((S, S), float("-inf"), device=scores.device, dtype=scores.dtype),
                        diagonal=1,
                    )
                    causal_mask[:, past_len:] = causal_block
                    return scores + causal_mask.unsqueeze(0).unsqueeze(0)
                return scores
            mask_slice = attn_mask[..., :total_S]
            if mask_slice.dtype == torch.bool:
                return scores.masked_fill(~mask_slice, float("-inf"))
            return scores + mask_slice.to(scores.dtype)

        # D. Atención: 1) Absorción completa, 2) SDPA nativo (entrenamiento/prefill), 3) Ruta estándar con caché
        if use_weight_absorption:
            w_uk_heads = w_uk_eff.to(q_c.dtype).view(self.n_heads, self.qk_nope_head_dim, self.kv_lora_rank)
            w_uv_heads = w_uv_eff.to(q_c.dtype).view(self.n_heads, self.v_head_dim, self.kv_lora_rank)
            q_c_absorbed = torch.einsum("b h s d, h d c -> b h s c", q_c, w_uk_heads)
            scores_c = torch.matmul(q_c_absorbed, c_kv.unsqueeze(1).transpose(-1, -2))
            scores_pe = torch.matmul(q_pe, k_pe.unsqueeze(1).transpose(-1, -2))
            attn_scores = _apply_mask((scores_c + scores_pe) * self.scale)
            attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q_c.dtype)
            o_latent = torch.matmul(attn_weights, c_kv.unsqueeze(1))
            out = torch.einsum("b h s c, h v c -> b h s v", o_latent, w_uv_heads)
        elif kv_cache is None and attn_mask is None:
            w_uk_cast = w_uk_eff.to(c_kv.dtype)
            w_uv_cast = w_uv_eff.to(c_kv.dtype)
            k_c = F.linear(c_kv, w_uk_cast).view(B, total_S, self.n_heads, self.qk_nope_head_dim).transpose(1, 2)
            v = F.linear(c_kv, w_uv_cast).view(B, total_S, self.n_heads, self.v_head_dim).transpose(1, 2)
            k_pe_heads = k_pe.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
            q_all = torch.cat([q_c, q_pe], dim=-1)
            k_all = torch.cat([k_c, k_pe_heads], dim=-1)
            out = F.scaled_dot_product_attention(q_all, k_all, v, is_causal=(S > 1), scale=self.scale)
        else:
            w_uk_cast = w_uk_eff.to(c_kv.dtype)
            w_uv_cast = w_uv_eff.to(c_kv.dtype)
            k_c = F.linear(c_kv, w_uk_cast).view(B, total_S, self.n_heads, self.qk_nope_head_dim).transpose(1, 2)
            v = F.linear(c_kv, w_uv_cast).view(B, total_S, self.n_heads, self.v_head_dim).transpose(1, 2)
            scores_c = torch.matmul(q_c, k_c.transpose(-1, -2))
            scores_pe = torch.matmul(q_pe, k_pe.unsqueeze(1).transpose(-1, -2))
            attn_scores = _apply_mask((scores_c + scores_pe) * self.scale)
            attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q_c.dtype)
            out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).contiguous().view(B, S, self.n_heads * self.v_head_dim)
        return self.w_out(out, add_sub_mode=add_sub_mode), new_kv_cache
