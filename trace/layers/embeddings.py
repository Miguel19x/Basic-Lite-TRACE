# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available

import math

import torch
import torch.nn as nn


class SinusoidalStepEncoding(nn.Module):
    """
    Codificación sinusoidal continua y sin parámetros huérfanos para el progreso recursivo
    normalizado u_r = (r + 1) / R en (0, 1].
    Evita recrear tensores desde listas Python en cada paso para favorecer captura de grafo.
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, f"SinusoidalStepEncoding requiere dim par, recibido: {dim}"
        half = dim // 2
        freqs = torch.exp(torch.arange(half, dtype=torch.float32) * (-math.log(100.0) / max(1, half - 1)))
        self.register_buffer("freqs", freqs.view(1, half), persistent=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, u_norm: float, device: torch.device) -> torch.Tensor:
        angles = (u_norm * math.pi) * self.freqs
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return self.proj(emb).view(1, 1, -1)
