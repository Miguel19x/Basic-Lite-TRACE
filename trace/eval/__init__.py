# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Evaluación de 4 dominios, matriz simétrica C1-C3 y calibración Brier/ECE (Fase 4B)."""

from trace.eval.bench_4domains import (
    BenchmarkTask,
    GenerationResult,
    ModelBackend,
    build_default_l3_corpus,
    build_default_tasks,
    build_extended_l3_corpus,
    build_extended_suite,
    compute_response_brier,
    compute_token_ece,
    count_think_tokens,
    evaluate_4domains,
    grade_answer,
)

__all__ = [
    "BenchmarkTask",
    "GenerationResult",
    "ModelBackend",
    "build_default_l3_corpus",
    "build_default_tasks",
    "build_extended_l3_corpus",
    "build_extended_suite",
    "compute_response_brier",
    "compute_token_ece",
    "count_think_tokens",
    "evaluate_4domains",
    "grade_answer",
]
