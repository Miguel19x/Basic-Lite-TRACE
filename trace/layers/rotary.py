# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Codificación posicional rotatoria (RoPE) con caché persistente y soporte para start_pos."""
    def __init__(self, dim: int, max_seq_len: int = 128):
        super().__init__()
        assert dim % 2 == 0, f"RotaryEmbedding requiere dim par, recibido: {dim}"
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None, :, :], persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        seq_len = x.size(2)
        assert 0 <= start_pos and start_pos + seq_len <= self.cos.size(2), (
            f"Posición RoPE fuera de rango: start_pos={start_pos}, seq_len={seq_len}, max={self.cos.size(2)}"
        )
        orig_dtype = x.dtype
        x_f32 = x.float()
        cos = self.cos[:, :, start_pos : start_pos + seq_len, :].to(device=x.device, dtype=torch.float32)
        sin = self.sin[:, :, start_pos : start_pos + seq_len, :].to(device=x.device, dtype=torch.float32)
        return ((x_f32 * cos) + (self._rotate_half(x_f32) * sin)).to(orig_dtype)
