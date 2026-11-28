# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Tests unitarios de Fase 5: generador SFT <think> + TIR verificable y recompensas GRPO."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from trace.eval.bench_4domains import _extract_json_object, grade_answer
from trace.reasoning.dataset import build_verified_tir_dataset, verify_no_bench4b_leakage
from trace.reasoning.rewards import compute_grpo_advantages, score_reasoning_completion
from trace.reasoning.trainer import (
    enable_bitnet_qat_ste_on_lora,
    gather_token_logprobs,
    grpo_policy_loss,
    quantize_w158_ste,
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


def test_quantize_w158_ste_produces_exact_ternary_levels():
    torch.manual_seed(3)
    w = torch.randn(16, 16, requires_grad=True)
    w_q = quantize_w158_ste(w)
    scale = w.detach().abs().mean().clamp_min(1e-5)
    codes = torch.round(w_q / scale)
    assert set(codes.unique().tolist()).issubset({-1.0, 0.0, 1.0})
    w_q.sum().backward()
    assert torch.allclose(w.grad, torch.ones_like(w))


def test_enable_bitnet_qat_ste_on_lora_propagates_gradients():
    torch.manual_seed(5)

    class _FakePeftLora(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_layer = nn.Linear(8, 8, bias=False)
            self.base_layer.weight.requires_grad_(False)
            self.lora_A = nn.ModuleDict({"default": nn.Linear(8, 2, bias=False)})
            self.lora_B = nn.ModuleDict({"default": nn.Linear(2, 8, bias=False)})
            nn.init.ones_(self.lora_B["default"].weight)
            self.scaling = {"default": 2.0}
            self.active_adapters = ["default"]
            self.disable_adapters = False
            self.merged = False

    mod = _FakePeftLora()
    assert enable_bitnet_qat_ste_on_lora(mod) == 1
    x = torch.randn(2, 8)
    out = mod(x)
    out.sum().backward()
    assert float(mod.lora_A["default"].weight.grad.abs().sum().item()) > 0.0
    assert float(mod.lora_B["default"].weight.grad.abs().sum().item()) > 0.0


def test_verified_tir_dataset_is_bilingual_50_50_en_es():
    examples = build_verified_tir_dataset(
        num_math=20,
        num_code=12,
        num_factual=12,
        general_ratio=0.15,
        seed=42,
    )
    en_count = sum(1 for ex in examples if ex.language == "en")
    es_count = sum(1 for ex in examples if ex.language == "es")
    assert en_count + es_count == len(examples)
    assert abs(en_count - es_count) <= 2


def test_verified_tir_dataset_uses_english_pivoted_thoughts_with_varied_phrasings():
    examples = build_verified_tir_dataset(
        num_math=20,
        num_code=12,
        num_factual=12,
        general_ratio=0.15,
        seed=42,
    )
    math_thoughts = [ex.thought_before_tool for ex in examples if ex.category == "math_tir"]
    # Multiple distinct phrasing stems (not 1 monolithic repeated template)
    distinct_prefixes = {t[:28] for t in math_thoughts}
    assert len(distinct_prefixes) >= 3
    # English-pivoted thinking (no Spanish-only boilerplate)
    assert all("invoco" not in t.lower() and "propenso" not in t.lower() for t in math_thoughts)


def test_verified_tir_dataset_includes_bilingual_abstention_examples():
    examples = build_verified_tir_dataset(
        num_math=20,
        num_code=12,
        num_factual=12,
        general_ratio=0.15,
        seed=42,
    )
    answers = {ex.answer for ex in examples}
    assert "INSUFFICIENT_EVIDENCE" in answers
    assert "EVIDENCIA_INSUFICIENTE" in answers
    assert verify_no_bench4b_leakage(examples) is True


def test_score_reasoning_completion_does_not_penalize_deep_non_repetitive_thinking():
    deep_steps = " ".join(f"Step {i}: verify intermediate quantity q{i} equals {i * 7}." for i in range(1, 22))
    text = f"<think>{deep_steps}</think>\n<answer>147</answer>"
    bd = score_reasoning_completion(text, "147")
    assert bd.outcome_reward == 1.0
    assert bd.length_penalty == 0.0
    assert bd.repetition_penalty == 0.0
    assert bd.total_reward >= 1.10


def test_score_reasoning_completion_penalizes_degenerate_loop_in_think():
    loop_think = "check da915fc77e vs da915fc77e no it is da915fc77e " * 5
    text = f"<think>{loop_think}</think>\n<answer>42</answer>"
    bd = score_reasoning_completion(text, "42")
    assert bd.outcome_reward == 1.0
    assert bd.repetition_penalty > 0.0
    # Correct answer still strictly beats a short wrong guess
    short_wrong = score_reasoning_completion("<think>Guess.</think>\n<answer>99</answer>", "42")
    assert bd.total_reward > short_wrong.total_reward


def test_verified_tir_dataset_generates_bilingual_math_word_problems():
    examples = build_verified_tir_dataset(num_math=24, num_code=4, num_factual=4, general_ratio=0.15, seed=42)
    math_en = [ex.prompt for ex in examples if ex.category == "math_tir" and ex.language == "en"]
    math_es = [ex.prompt for ex in examples if ex.category == "math_tir" and ex.language == "es"]
    assert any("batch job processes" in p for p in math_en)
    assert any("Un proceso gestiona" in p for p in math_es)


def test_score_reasoning_completion_does_not_match_tool_call_args_as_answer():
    only_tool = '<think>Use calc.</think>\n<tool_call>{"tool":"calc","args":{"expr":"42 + 1"}}</tool_call>'
    bd = score_reasoning_completion(only_tool, "42")
    assert bd.outcome_reward == 0.0
    assert bd.valid_tool_json is True


def test_kaggle_notebooks_gate_intermediate_cleanup_on_export_success():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for nb_name in (
        "fase5_bitnet_2b_think_tir_kaggle.ipynb",
        "fase5_bitnet_2b_think_tir_tpu_v5e8_kaggle.ipynb",
    ):
        nb = json.loads((root / "notebooks" / nb_name).read_text(encoding="utf-8"))
        export_cells = [
            "".join(c.get("source", []))
            for c in nb["cells"]
            if c.get("cell_type") == "code" and "import_bitnet.py" in "".join(c.get("source", []))
        ]
        assert len(export_cells) == 1
        src = export_cells[0]
        assert "!rm -rf" not in src
        assert "&& test -s" in src
        assert "&& rm -rf" in src


