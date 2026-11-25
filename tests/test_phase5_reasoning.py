# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Tests unitarios de Fase 5: generador SFT <think> + TIR verificable y recompensas GRPO."""

from __future__ import annotations

import pytest
import torch

from trace.eval.bench_4domains import _extract_json_object, grade_answer
from trace.reasoning.dataset import build_verified_tir_dataset, verify_no_bench4b_leakage
from trace.reasoning.rewards import compute_grpo_advantages, score_reasoning_completion
from trace.reasoning.trainer import (
    gather_token_logprobs,
    grpo_policy_loss,
    sft_loss_on_completion,
)


def test_verified_tir_dataset_enforces_general_mix_ratio():
    examples = build_verified_tir_dataset(
        num_math=10,
        num_code=6,
        num_factual=4,
        general_ratio=0.15,
        seed=42,
    )
    general_count = sum(1 for ex in examples if ex.category == "general_no_tool")
    assert general_count / len(examples) >= 0.10
    assert verify_no_bench4b_leakage(examples) is True


def test_verified_tir_dataset_rejects_low_general_ratio():
    with pytest.raises(ValueError):
        build_verified_tir_dataset(general_ratio=0.05)


def test_score_reasoning_completion_rewards_correct_structured_trace():
    text = (
        "<think>Calculo con calc.</think>\n"
        '<tool_call>{"tool":"calc","args":{"expr":"6*7"}}</tool_call>\n'
        "<answer>42</answer>"
    )
    bd = score_reasoning_completion(text, "42")
    assert bd.outcome_reward == 1.0
    assert bd.valid_tool_json is True
    assert bd.length_penalty == 0.0
    assert bd.total_reward == pytest.approx(1.20, abs=1e-6)


def test_score_reasoning_completion_penalizes_excessive_length():
    long_text = "<think>" + ("x" * 1000) + "</think>\n<answer>42</answer>"
    bd = score_reasoning_completion(long_text, "42", max_target_chars=200)
    assert bd.outcome_reward == 1.0
    assert bd.length_penalty > 0.0
    assert bd.total_reward < 1.10


def test_compute_grpo_advantages_zero_mean_for_mixed_group():
    adv = compute_grpo_advantages([1.2, 0.2, 0.2, 0.2])
    assert len(adv) == 4
    assert adv[0] > 0.0
    assert adv[1] < 0.0
    assert sum(adv) == pytest.approx(0.0, abs=1e-6)


def test_compute_grpo_advantages_returns_zeros_for_identical_rewards():
    assert compute_grpo_advantages([1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0]


def test_sft_loss_masks_prompt_tokens():
    torch.manual_seed(7)
    logits = torch.randn(1, 8, 16, requires_grad=True)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    prompt_lengths = torch.tensor([5], dtype=torch.long)

    loss = sft_loss_on_completion(logits, input_ids, prompt_lengths, pad_id=0)
    loss.backward()
    # Posiciones 0..3 predicen tokens 1..4 (todos del prompt): gradiente debe ser exactamente 0
    assert torch.all(logits.grad[0, :4, :] == 0.0)
    # Posiciones 4..6 predicen tokens 5..7 (completado): gradiente no nulo
    assert float(logits.grad[0, 4:7, :].abs().sum().item()) > 0.0


def test_grpo_policy_loss_kl_is_non_negative():
    torch.manual_seed(11)
    logits_new = torch.randn(2, 6, 12, requires_grad=True)
    logits_old = logits_new.detach().clone()
    logits_ref = torch.randn(2, 6, 12)
    ids = torch.randint(1, 12, (2, 6))
    mask = torch.ones(2, 5)
    adv = torch.tensor([1.0, -1.0])

    log_new = gather_token_logprobs(logits_new, ids)
    log_old = gather_token_logprobs(logits_old, ids)
    log_ref = gather_token_logprobs(logits_ref, ids)
    loss, metrics = grpo_policy_loss(log_new, log_old, log_ref, adv, mask, kl_beta=0.05)
    loss.backward()
    assert metrics["kl_mean"] >= 0.0
    assert float(logits_new.grad.abs().sum().item()) > 0.0


def test_bench4b_extracts_phase5_tool_call_tags():
    raw = '<think>uso calc</think>\n<tool_call>{"tool":"calc","args":{"expr":"2+2"}}</tool_call>'
    assert _extract_json_object(raw) == {"tool": "calc", "args": {"expr": "2+2"}}


def test_bench4b_grade_answer_prefers_answer_tag():
    resp = "<think>El año incorrecto sería 1542, pero L3 dice 1494.</think>\n<answer>1494</answer>"
    assert grade_answer(resp, "1494") is True
    assert grade_answer(resp, "1542") is False


def test_bench4b_grade_answer_rejects_negative_or_decimal_mismatch():
    assert grade_answer("El resultado es -70", "70") is False
    assert grade_answer("El resultado es 70.5", "70") is False


def test_sft_loss_respects_explicit_tool_response_loss_mask():
    torch.manual_seed(13)
    logits = torch.randn(1, 8, 16, requires_grad=True)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 2]], dtype=torch.long)
    prompt_lengths = torch.tensor([3], dtype=torch.long)
    # Enmascarar posiciones 4..5 (simula <tool_response>) manteniendo EOS final en pos 7 activo
    loss_mask = torch.tensor([[0, 0, 0, 1, 0, 0, 1, 1]], dtype=torch.float32)

    loss = sft_loss_on_completion(logits, input_ids, prompt_lengths, pad_id=2, loss_mask=loss_mask)
    loss.backward()
    # Posiciones 3 y 4 predicen índices 4 y 5 (enmascarados a 0): gradiente exacto 0
    assert torch.all(logits.grad[0, 3:5, :] == 0.0)
    # Posición 6 predice índice 7 (EOS final = 2 == pad_id): gradiente NO nulo gracias a loss_mask
    assert float(logits.grad[0, 6, :].abs().sum().item()) > 0.0
