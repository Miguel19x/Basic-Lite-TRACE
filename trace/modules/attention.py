# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available

from trace.layers.bitlinear import BitLinear
from trace.layers.norm import RMSNorm
from trace.layers.rotary import RotaryEmbedding

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_block_causal_mask(seq_len: int, block_size: int, device: torch.device) -> torch.Tensor:
    """Construye máscara causal bidireccional local dentro del bloque y causal estricta hacia bloques pasados."""
    idx = torch.arange(seq_len, device=device)
    block_id = idx // block_size
    allowed = block_id.unsqueeze(0) <= block_id.unsqueeze(1)
    return allowed.unsqueeze(0).unsqueeze(0)


def build_target_unified_mask(
    prompt_lens: list[int],
    max_seq_len: int,
    block_size: int,
    device: torch.device,
    target_len: int = 35,
) -> torch.Tensor:
    """
    Construye máscara causal con región de target unificada [B, 1, L, L]:
    1. Región de Prompt (0..plen-1): causalidad estricta por bloques (block_size=16).
       Los tokens del prompt nunca pueden atender a la región de respuesta (target).
    2. Región de Target (plen..plen+target_len-1): bloque unificado bidireccional.
       Cualquier dígito de respuesta (centenas, decenas, unidades de pw, pms, watt, ms)
       puede atender bidireccionalmente a todos los demás dígitos del target
       y a todos los tokens del prompt previo, eliminando el corte causal de acarreos.
    3. Región post-target (PAD): no atiende hacia adelante.
    """
    B = len(prompt_lens)
    L = max_seq_len
    idx = torch.arange(L, device=device)
    block_id = idx // block_size

    base_mask = (block_id.unsqueeze(0) <= block_id.unsqueeze(1)).unsqueeze(0).repeat(B, 1, 1)

    for b in range(B):
        plen = prompt_lens[b]
        t_end = min(L, plen + target_len)
        base_mask[b, :plen, plen:] = False
        base_mask[b, plen:t_end, :t_end] = True
        base_mask[b, plen:t_end, t_end:] = False

    return base_mask.unsqueeze(1)


class BlockCausalGQATernaryAttention(nn.Module):
    """
    Atención Grouped-Query (GQA 4:2) cuantizada en pesos ternarios W1.58A8 con:
    - Normalización RMSNorm en Q y K (estilo Qwen 2.5 / Gemma 2).
    - Incrustación posicional RoPE con soporte start_pos.
    - Máscara Block-Causal (CDLM) y compatibilidad con caché KV inmutable.
    """
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.num_queries_per_kv = cfg.n_heads // cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.kv_dim = self.n_kv_heads * self.head_dim
        self.block_size = cfg.block_size

        self.q_proj = BitLinear(cfg.d_model, cfg.d_model, eps=cfg.eps)
        self.k_proj = BitLinear(cfg.d_model, self.kv_dim, eps=cfg.eps)
        self.v_proj = BitLinear(cfg.d_model, self.kv_dim, eps=cfg.eps)
        self.o_proj = BitLinear(cfg.d_model, cfg.d_model, eps=cfg.eps)
        self.rope = RotaryEmbedding(self.head_dim, cfg.max_seq_len)
        self.use_qk_norm = getattr(cfg, "use_qk_norm", True)
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=cfg.eps)
            self.k_norm = RMSNorm(self.head_dim, eps=cfg.eps)
        self._mask_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    def _get_block_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (seq_len, device)
        if key not in self._mask_cache:
            self._mask_cache[key] = build_block_causal_mask(seq_len, self.block_size, device)
        return self._mask_cache[key]

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_queries_per_kv == 1:
            return x
        B, H_kv, L, D_h = x.shape
        return (
            x[:, :, None, :, :]
            .expand(B, H_kv, self.num_queries_per_kv, L, D_h)
            .reshape(B, self.n_heads, L, D_h)
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        start_pos: int = 0,
        record_telemetry: bool = False,
        integer_mode: bool = False,
        add_sub_mode: bool = False,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, L, D = x.shape
        q = self.q_proj(x, record_telemetry, integer_mode, add_sub_mode).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x, record_telemetry, integer_mode, add_sub_mode).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x, record_telemetry, integer_mode, add_sub_mode).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = self.rope.apply_rope(q, start_pos=start_pos)
        k = self.rope.apply_rope(k, start_pos=start_pos)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k_full = torch.cat([k_prev, k], dim=2)
            v_full = torch.cat([v_prev, v], dim=2)
        else:
            k_full, v_full = k, v

        k_exp = self._repeat_kv(k_full)
        v_exp = self._repeat_kv(v_full)

        if kv_cache is not None:
            attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)
        else:
            eff_mask = attn_mask if attn_mask is not None else self._get_block_mask(L, x.device)
            attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=eff_mask)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.o_proj(attn_out, record_telemetry, integer_mode, add_sub_mode)
        return out, (k_full, v_full)
