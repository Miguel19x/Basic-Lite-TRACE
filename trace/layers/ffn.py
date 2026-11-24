# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available

from trace.layers.bitlinear import BitLinear

import torch
import torch.nn as nn
import torch.nn.functional as F


class TernaryExpert(nn.Module):
    """Experto individual con proyecciones ternarias BitLinear y activación Squared-ReLU."""

    def __init__(self, d_model: int, d_ff: int, eps: float = 1e-6):
        super().__init__()
        # Conserva estrictamente los nombres 'self.up' y 'self.down' para compatibilidad 1:1 con checkpoints Fase 1-2
        self.up = BitLinear(d_model, d_ff, eps=eps)
        self.down = BitLinear(d_ff, d_model, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        record_telemetry: bool = False,
        integer_mode: bool = False,
        add_sub_mode: bool = False,
    ) -> torch.Tensor:
        # Activación Squared-ReLU exacta de Fase 1-2: F.relu(...).square()
        h = F.relu(self.up(x, record_telemetry, integer_mode, add_sub_mode)).square()
        return self.down(h, record_telemetry, integer_mode, add_sub_mode)
