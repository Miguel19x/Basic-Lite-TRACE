# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available

from dataclasses import dataclass, field

import torch


@dataclass
class ModelOutput:
    """
    Salida estructurada para el modelo TernaryRecursiveDiffusionLM.
    Permite desacoplar proyecciones de vocabulario, estados latentes y telemetría.
    """
    final_logits: torch.Tensor
    intermediate_logits: list[torch.Tensor] = field(default_factory=list)
    halt_probs: list[torch.Tensor] = field(default_factory=list)
    kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)
    carried_states: tuple[torch.Tensor, torch.Tensor] | None = None
    last_hidden_state: torch.Tensor | None = None

    # Compatibilidad con desempaquetado de tupla clásico:
    # final_logits, intermediate_logits, halt_probs, kv_caches, carried_states = model(...)
    def __iter__(self):
        yield self.final_logits
        yield self.intermediate_logits
        yield self.halt_probs
        yield self.kv_caches
        yield self.carried_states

    def __getitem__(self, idx):
        return tuple(self)[idx]
