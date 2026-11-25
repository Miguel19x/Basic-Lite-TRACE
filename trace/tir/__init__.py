# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Capa TIR (Tool-Integrated Reasoning): sandbox de permisos, enciclopedia SQLite L3 y registro de herramientas."""

from trace.tir.l3_store import L3Encyclopedia
from trace.tir.sandbox import Sandbox, SandboxPolicy, SandboxViolationError
from trace.tir.tools import TIRToolRegistry, safe_eval_math

__all__ = [
    "L3Encyclopedia",
    "Sandbox",
    "SandboxPolicy",
    "SandboxViolationError",
    "TIRToolRegistry",
    "safe_eval_math",
]
