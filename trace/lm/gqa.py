# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Oráculo PyTorch para modelos BitNet W1.58 con GQA, RoPE split-half y FFN ReLU² con compuerta."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from trace.layers.bitlinear import BitLinear
from trace.layers.norm import RMSNorm
from trace.lm.config import TRACELMConfig


def _apply_rope_split_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE estilo LLaMA/BitNet: pares (i, i + head_dim // 2). x: [B, H, S, D], cos/sin: [1, 1, S, D//2]."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def _apply_rope_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE intercalado: pares (2i, 2i+1). x: [B, H, S, D], cos/sin: [1, 1, S, D//2]."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x2 * cos + x1 * sin
    return out


def _quantize_kv_int8_per_head(t: torch.Tensor) -> torch.Tensor:
    """Simula cuantización/descuantización INT8 simétrica por token y cabeza KV."""
    amax = t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    scale = amax / 127.0
    scale_inv = 127.0 / amax
    q = torch.round(t * scale_inv).clamp(-127.0, 127.0)
    return q * scale


class BitNetGQAAttention(nn.Module):
    """Atención Grouped-Query (GQA) compatible con microsoft/bitnet-b1.58-2B-4T."""

    def __init__(self, cfg: TRACELMConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads if cfg.n_kv_heads > 0 else cfg.n_heads
        self.head_dim = cfg.head_dim if cfg.head_dim > 0 else (cfg.d_model // cfg.n_heads)
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) debe ser múltiplo de n_kv_heads ({self.n_kv_heads})")
        self.group_size = self.n_heads // self.n_kv_heads
        self.rope_interleaved = cfg.rope_interleaved
        self.scale = 1.0 / math.sqrt(self.head_dim)

        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim

        # BitNet-2B-4T: Q/K/V sin sub_ln propio (toman attn_norm), o_proj con attn_sub_norm (= w_out.sub_ln)
        self.w_q = BitLinear(cfg.d_model, q_dim, eps=cfg.eps, use_sub_ln=False, quantize_weights_enabled=cfg.quantize_weights_enabled, weight_scale_type=cfg.weight_scale_type)
        self.w_k = BitLinear(cfg.d_model, kv_dim, eps=cfg.eps, use_sub_ln=False, quantize_weights_enabled=cfg.quantize_weights_enabled, weight_scale_type=cfg.weight_scale_type)
        self.w_v = BitLinear(cfg.d_model, kv_dim, eps=cfg.eps, use_sub_ln=False, quantize_weights_enabled=cfg.quantize_weights_enabled, weight_scale_type=cfg.weight_scale_type)
        self.w_out = BitLinear(q_dim, cfg.d_model, eps=cfg.eps, use_sub_ln=True, quantize_weights_enabled=cfg.quantize_weights_enabled, weight_scale_type=cfg.weight_scale_type)

    def forward(
        self,
        h: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        quantize_kv_int8: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, S, _ = h.shape
        q = self.w_q(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(h).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(h).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.rope_interleaved:
            q = _apply_rope_interleaved(q, cos, sin)
            k = _apply_rope_interleaved(k, cos, sin)
        else:
            q = _apply_rope_split_half(q, cos, sin)
            k = _apply_rope_split_half(k, cos, sin)

        if quantize_kv_int8:
            k = _quantize_kv_int8_per_head(k)
            v = _quantize_kv_int8_per_head(v)

        if kv_cache is not None:
            k_past, v_past = kv_cache
            k = torch.cat([k_past, k], dim=2)
            v = torch.cat([v_past, v], dim=2)
        new_cache = (k, v)

        total_len = k.shape[2]
        k_rep = k.repeat_interleave(self.group_size, dim=1)
        v_rep = v.repeat_interleave(self.group_size, dim=1)

        scores = torch.matmul(q, k_rep.transpose(-2, -1)) * self.scale
        if S > 1:
            mask = torch.triu(
                torch.full((S, total_len), float("-inf"), device=h.device, dtype=scores.dtype),
                diagonal=total_len - S + 1,
            )
            scores = scores + mask
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v_rep).transpose(1, 2).contiguous().view(B, S, self.n_heads * self.head_dim)
        return self.w_out(out), new_cache


class BitNetGQAFFN(nn.Module):
    """FFN con compuerta ReLU²(gate) * up + ffn_sub_norm (= down.sub_ln)."""

    def __init__(self, cfg: TRACELMConfig):
        super().__init__()
        self.use_gated = cfg.use_gated_ffn
        if self.use_gated:
            self.gate = BitLinear(cfg.d_model, cfg.d_ff, eps=cfg.eps, use_sub_ln=False, quantize_weights_enabled=cfg.quantize_weights_enabled, weight_scale_type=cfg.weight_scale_type)
        self.up = BitLinear(cfg.d_model, cfg.d_ff, eps=cfg.eps, use_sub_ln=False, quantize_weights_enabled=cfg.quantize_weights_enabled, weight_scale_type=cfg.weight_scale_type)
        self.down = BitLinear(cfg.d_ff, cfg.d_model, eps=cfg.eps, use_sub_ln=True, quantize_weights_enabled=cfg.quantize_weights_enabled, weight_scale_type=cfg.weight_scale_type)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.use_gated:
            g = F.relu(self.gate(h)) ** 2
            return self.down(g * self.up(h))
        return self.down(F.relu(self.up(h)) ** 2)


class BitNetGQALayer(nn.Module):
    def __init__(self, cfg: TRACELMConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.eps)
        self.attn = BitNetGQAAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, eps=cfg.eps)
        self.ffn = BitNetGQAFFN(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        quantize_kv_int8: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_cache = self.attn(self.attn_norm(x), cos, sin, kv_cache=kv_cache, quantize_kv_int8=quantize_kv_int8)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


class BitNetGQALM(nn.Module):
    """Modelo de lenguaje BitNet W1.58 con GQA serializable a Paged BitCOS v1.1 (trace-lm-v1)."""

    def __init__(self, cfg: TRACELMConfig):
        super().__init__()
        if cfg.attn_type != "gqa":
            raise ValueError(f"BitNetGQALM requiere attn_type='gqa', recibido {cfg.attn_type!r}")
        self.cfg = cfg
        self.head_dim = cfg.head_dim if cfg.head_dim > 0 else (cfg.d_model // cfg.n_heads)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([BitNetGQALayer(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        # Precomputar tablas RoPE [max_seq_len, head_dim // 2]
        half = self.head_dim // 2
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / float(self.head_dim)))
        positions = torch.arange(cfg.max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)
        self.register_buffer("rope_cos", torch.cos(angles), persistent=False)
        self.register_buffer("rope_sin", torch.sin(angles), persistent=False)
        assert self.rope_cos.shape == (cfg.max_seq_len, half)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_kvs: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        quantize_kv_int8: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        _, S = input_ids.shape
        past_len = past_kvs[0][0].shape[2] if past_kvs is not None else 0
        cos = self.rope_cos[past_len : past_len + S].unsqueeze(0).unsqueeze(0)
        sin = self.rope_sin[past_len : past_len + S].unsqueeze(0).unsqueeze(0)

        x = self.tok_emb(input_ids)
        new_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for idx, layer in enumerate(self.layers):
            layer_kv = past_kvs[idx] if past_kvs is not None else None
            x, new_kv = layer(x, cos, sin, kv_cache=layer_kv, quantize_kv_int8=quantize_kv_int8)
            new_kvs.append(new_kv)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, new_kvs
