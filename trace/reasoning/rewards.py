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

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_THINK_UNCLOSED_RE = re.compile(r"<think>(.*)$", re.DOTALL | re.IGNORECASE)
_TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
_ANS_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_CHAR_RUN_RE = re.compile(r"(.)\1{15,}")


@dataclass(frozen=True)
class GRPORewardBreakdown:
    outcome_reward: float
    format_reward: float
    length_penalty: float
    total_reward: float
    extracted_answer: str
    valid_tool_json: bool
    repetition_penalty: float = 0.0


def compute_degenerate_loop_penalty(completion: str) -> float:
    """
    Detecta bucles degenerados de repetición dentro de `<think>` sin penalizar jamás
    el pensamiento largo que progresa con pasos distintos hacia la solución:
      - Secuencias repetidas de >= 16 caracteres idénticos seguidos.
      - 5-gramas exactos (5 palabras) repetidos >= 3 veces dentro del bloque `<think>`.
    Tope estricto 0.25 para que una respuesta correcta (`outcome=1.0`) supere siempre
    a una respuesta corta incorrecta (`outcome=0.0`).
    """
    blocks = _THINK_RE.findall(completion)
    stripped = _THINK_RE.sub("", completion)
    unclosed = _THINK_UNCLOSED_RE.search(stripped)
    if unclosed is not None:
        blocks.append(unclosed.group(1))
    think_text = " ".join(blocks).strip()
    if not think_text:
        return 0.0

    penalty = 0.0
    if _CHAR_RUN_RE.search(think_text) is not None:
        penalty = max(penalty, 0.20)

    words = think_text.lower().split()
    if len(words) >= 15:
        counts: dict[tuple[str, ...], int] = {}
        for i in range(len(words) - 4):
            ngram = tuple(words[i : i + 5])
            counts[ngram] = counts.get(ngram, 0) + 1
        max_rep = max(counts.values())
        if max_rep >= 3:
            penalty = max(penalty, min(0.25, 0.05 * (max_rep - 2)))
    return round(min(0.25, penalty), 6)


def score_reasoning_completion(
    completion: str,
    expected_answer: str,
    *,
    max_target_chars: int = 2048,
) -> GRPORewardBreakdown:
    """
    Recompensa verificable (RLVR) sin modelo de recompensa neuronal:
      - outcome_reward: +1.0 si `<answer>` (o texto final) coincide exactamente con `expected_answer`, 0.0 si no.
      - format_reward: +0.20 distribuido en `<think>`, JSON válido en `<tool_call>` y `<answer>`.
      - repetition_penalty: penaliza solo bucles degenerados en `<think>` (hasta 0.25), nunca pensamiento útil.
      - length_penalty: techo generoso (`max_target_chars=2048` por defecto) para no truncar razonamiento profundo.
    """
    has_think = _THINK_RE.search(completion) is not None
    cleaned = _THINK_RE.sub("", completion)
    cleaned = _THINK_UNCLOSED_RE.sub("", cleaned)
    cleaned = _TOOL_RE.sub("", cleaned)
    ans_matches = _ANS_RE.findall(cleaned)
    has_answer = len(ans_matches) > 0
    extracted = ans_matches[-1].strip() if has_answer else cleaned.strip()

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
    repetition_penalty = compute_degenerate_loop_penalty(completion)

    total = outcome_reward + format_reward - length_penalty - repetition_penalty
    return GRPORewardBreakdown(
        outcome_reward=outcome_reward,
        format_reward=round(format_reward, 6),
        length_penalty=round(length_penalty, 6),
        total_reward=round(total, 6),
        extracted_answer=extracted,
        valid_tool_json=valid_tool_json,
        repetition_penalty=repetition_penalty,
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
