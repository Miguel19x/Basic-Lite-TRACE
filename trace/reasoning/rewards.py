# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Recompensas verificables RLVR y cálculo de ventajas GRPO para Fase 5."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

from trace.eval.bench_4domains import grade_answer

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_ANS_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


@dataclass(frozen=True)
class GRPORewardBreakdown:
    outcome_reward: float
    format_reward: float
    length_penalty: float
    total_reward: float
    extracted_answer: str
    valid_tool_json: bool


def score_reasoning_completion(
    completion: str,
    expected_answer: str,
    *,
    max_target_chars: int = 640,
) -> GRPORewardBreakdown:
    """
    Recompensa verificable (RLVR) sin modelo de recompensa neuronal:
      - outcome_reward: +1.0 si `<answer>` (o texto final) coincide exactamente con `expected_answer`, 0.0 si no.
      - format_reward: +0.20 distribuido en `<think>`, JSON válido en `<tool_call>` y `<answer>`.
      - length_penalty: penalización suave si excede `max_target_chars` (control estricto de longitud en CPU).
    """
    has_think = _THINK_RE.search(completion) is not None
    ans_match = _ANS_RE.search(completion)
    has_answer = ans_match is not None
    extracted = ans_match.group(1).strip() if ans_match else completion.strip()

    tool_match = _TOOL_RE.search(completion)
    valid_tool_json = False
    if tool_match is not None:
        try:
            obj = json.loads(tool_match.group(1).strip())
            valid_tool_json = isinstance(obj, dict) and "tool" in obj and "args" in obj
        except json.JSONDecodeError:
            valid_tool_json = False

    outcome_reward = 1.0 if grade_answer(extracted, expected_answer) else 0.0

    format_reward = 0.0
    if has_think:
        format_reward += 0.05
    if valid_tool_json:
        format_reward += 0.10
    if has_answer:
        format_reward += 0.05

    overflow = max(0, len(completion) - max_target_chars)
    length_penalty = min(0.25, 0.0005 * overflow) if overflow > 0 else 0.0

    total = outcome_reward + format_reward - length_penalty
    return GRPORewardBreakdown(
        outcome_reward=outcome_reward,
        format_reward=round(format_reward, 6),
        length_penalty=round(length_penalty, 6),
        total_reward=round(total, 6),
        extracted_answer=extracted,
        valid_tool_json=valid_tool_json,
    )


def compute_grpo_advantages(rewards: list[float], eps: float = 1e-6) -> list[float]:
    """Ventajas normalizadas por grupo GRPO: A_i = (r_i - mean(r)) / (std(r) + eps)."""
    if not rewards:
        return []
    n = len(rewards)
    mean_r = sum(rewards) / n
    var_r = sum((r - mean_r) ** 2 for r in rewards) / n
    std_r = math.sqrt(var_r)
    if std_r < eps:
        return [0.0 for _ in rewards]
    return [(r - mean_r) / (std_r + eps) for r in rewards]
