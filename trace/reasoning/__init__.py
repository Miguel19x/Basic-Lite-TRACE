# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Fase 5: Razonamiento temprano (<think> + TIR verificable + recompensas GRPO/RLVR)."""

from trace.reasoning.dataset import (
    ReasoningTraceExample,
    build_verified_tir_dataset,
    format_trace_target,
    verify_no_bench4b_leakage,
)
from trace.reasoning.rewards import (
    GRPORewardBreakdown,
    compute_grpo_advantages,
    score_reasoning_completion,
)
from trace.reasoning.trainer import (
    gather_token_logprobs,
    grpo_policy_loss,
    score_group_completions,
    sft_loss_on_completion,
)

__all__ = [
    "GRPORewardBreakdown",
    "ReasoningTraceExample",
    "build_verified_tir_dataset",
    "compute_grpo_advantages",
    "format_trace_target",
    "gather_token_logprobs",
    "grpo_policy_loss",
    "score_group_completions",
    "score_reasoning_completion",
    "sft_loss_on_completion",
    "verify_no_bench4b_leakage",
]
