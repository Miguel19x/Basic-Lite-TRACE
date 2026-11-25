# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Tests de Fase 4B.3: Banco de 4 dominios, matriz simétrica C1-C3 y calibración ECE/Brier."""

from __future__ import annotations

import json
import re

import pytest

from trace.eval.bench_4domains import (
    GenerationResult,
    build_default_tasks,
    compute_response_brier,
    compute_token_ece,
    evaluate_4domains,
    grade_answer,
)


class _DeterministicStubBackend:
    """Backend determinista para validar C1 < C2 < C3 sin ejecutar modelos 2B en pytest."""

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        json_constrained: bool = False,
    ) -> GenerationResult:
        if json_constrained:
            if "sum_squares.py" in prompt:
                call = {"tool": "run_cmd", "args": {"argv": ["python3", "sum_squares.py"]}}
            elif "config_target.txt" in prompt:
                call = {"tool": "read_file", "args": {"path": "config_target.txt"}}
            elif "347 * 829" in prompt:
                call = {"tool": "calc", "args": {"expr": "(347 * 829) - (193 * 412) + (7 ** 4)"}}
            elif "gcd(4620, 3570)" in prompt:
                call = {"tool": "calc", "args": {"expr": "gcd(4620, 3570) * 19 + factorial(6)"}}
            elif "2400 metros" in prompt:
                call = {"tool": "calc", "args": {"expr": "85 - 15"}}
            else:
                call = {"tool": "none", "args": {}}
            return GenerationResult(
                text=json.dumps(call),
                token_confidences=[0.95],
                token_correctness=[1],
                response_confidence=0.95,
                tokens_generated=12,
            )

        # Si hay resultado de herramienta TIR en el prompt, extraerlo directamente
        if "Resultado de herramienta" in prompt:
            m_stdout = re.search(r'"stdout":\s*"(\d+)', prompt)
            if m_stdout:
                return GenerationResult(text=m_stdout.group(1), response_confidence=0.95, tokens_generated=4)
            m_port = re.search(r"TARGET_PORT=(\d+)", prompt)
            if m_port:
                return GenerationResult(text=m_port.group(1), response_confidence=0.95, tokens_generated=4)
            m_res = re.search(r'"result":\s*(\d+)', prompt)
            if m_res:
                return GenerationResult(text=m_res.group(1), response_confidence=0.95, tokens_generated=4)

        # Si hay contexto L3 (C2/C3), responder hechos de historia/geografía
        if "Contexto L3 (offline):" in prompt:
            if "1494" in prompt:
                return GenerationResult(text="1494", response_confidence=0.9, tokens_generated=4)
            if "1642" in prompt:
                return GenerationResult(text="1642", response_confidence=0.9, tokens_generated=4)

        # Localización aguja disponible desde C1
        if "DELTA-7492" in prompt:
            return GenerationResult(text="DELTA-7492", response_confidence=0.85, tokens_generated=4)

        # Sin contexto L3 ni herramientas, el stub falla
        return GenerationResult(
            text="desconocido",
            token_confidences=[0.6],
            token_correctness=[0],
            response_confidence=0.6,
            tokens_generated=4,
        )


def test_compute_token_ece_perfect_calibration_is_zero():
    ece = compute_token_ece([1.0, 1.0, 0.0, 0.0], [1, 1, 0, 0], n_bins=5)
    assert ece == pytest.approx(0.0, abs=1e-9)


def test_compute_response_brier_matches_analytic_formula():
    # ((0.8 - 1)^2 + (0.2 - 0)^2) / 2 = (0.04 + 0.04) / 2 = 0.04
    brier = compute_response_brier([0.8, 0.2], [1, 0])
    assert brier == pytest.approx(0.04, abs=1e-9)


def test_grade_answer_rejects_accidental_substring_match():
    assert grade_answer("El resultado es 70.", "70") is True
    assert grade_answer("El resultado es 1700.", "70") is False


def test_default_tasks_cover_four_domains_with_needle_vs_synthesis_split():
    tasks = build_default_tasks()
    domains = {t.domain for t in tasks}
    assert domains == {"code_terminal", "math_logic", "long_docs", "factual"}
    long_subtasks = {t.subtask for t in tasks if t.domain == "long_docs"}
    assert long_subtasks == {"needle", "synthesis"}


def test_evaluate_4domains_symmetric_c1_c2_c3_progression():
    report = evaluate_4domains(_DeterministicStubBackend(), max_new_tokens=64)

    assert report["out_of_sandbox_executions"] == 0
    c1 = report["conditions"]["C1"]
    c2 = report["conditions"]["C2"]
    c3 = report["conditions"]["C3"]

    assert c1["long_docs_breakdown"]["needle"] == 1.0
    assert c1["long_docs_breakdown"]["synthesis"] == 0.0
    assert c1["by_domain"]["factual"] == 0.0
    assert c2["by_domain"]["factual"] == 1.0
    assert c3["by_domain"]["code_terminal"] == 1.0
    assert c3["by_domain"]["math_logic"] == 1.0
    assert c3["long_docs_breakdown"]["synthesis"] == 1.0
    assert c3["overall_em"] == 1.0
    assert c3["valid_json_tool_calls"] == c3["total_json_tool_calls"] == report["num_tasks"]
    assert c3["response_brier"] < c1["response_brier"]
