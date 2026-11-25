# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Tests unitarios de extract_abacus_ranks, compute_ece, evaluate_causal y BitLinear."""

import random

import pytest
import torch

from config import HybridConfig
from data import ProceduralCompositionalDataset
from eval import compute_ece, evaluate_causal
from trace.layers.abacus import AbacusEmbedding, extract_abacus_ranks
from trace.layers.bitlinear import BitLinear


def test_extract_abacus_ranks_handcrafted_decimal_places():
    # "pw: 087\n" -> delimitador=15 para no dígitos, [2, 1, 0] para "087"
    ids = torch.tensor([[ord(c) for c in "pw: 087\n"]], dtype=torch.long)
    ranks = extract_abacus_ranks(ids, delimiter_rank=15, max_rank=14, mask_token_id=257)
    assert ranks.tolist() == [[15, 15, 15, 15, 2, 1, 0, 15]]


def test_extract_abacus_ranks_treats_mask_token_as_digit():
    # "x:" + [MASK, MASK, MASK] + "\n" -> [15, 15, 2, 1, 0, 15]
    ids = torch.tensor([[ord("x"), ord(":"), 257, 257, 257, ord("\n")]], dtype=torch.long)
    ranks = extract_abacus_ranks(ids, delimiter_rank=15, max_rank=14, mask_token_id=257)
    assert ranks.tolist() == [[15, 15, 2, 1, 0, 15]]


def test_extract_abacus_ranks_clamps_long_numbers_at_max_rank():
    ids = torch.tensor([[ord("9")] * 6], dtype=torch.long)
    ranks = extract_abacus_ranks(ids, delimiter_rank=15, max_rank=3, mask_token_id=None)
    assert ranks.tolist() == [[3, 3, 3, 2, 1, 0]]


def test_extract_abacus_ranks_invariant_property_sweep():
    rng = random.Random(19)
    alphabet = list("ab: \n0123456789")
    for _ in range(30):
        s = "".join(rng.choice(alphabet) for _ in range(24))
        ids = torch.tensor([[ord(c) for c in s]], dtype=torch.long)
        ranks = extract_abacus_ranks(ids, delimiter_rank=15, max_rank=14, mask_token_id=None)[0].tolist()
        for ch, rk in zip(s, ranks):
            if ch.isdigit():
                assert 0 <= rk <= 14
            else:
                assert rk == 15


def test_abacus_embedding_rejects_invalid_max_ranks():
    with pytest.raises(AssertionError):
        AbacusEmbedding(d_model=16, max_ranks=1)


def test_compute_ece_exact_hand_calculated_values():
    # Perfectamente calibrado: conf == acc en cada bin -> 0.0 %
    assert compute_ece([0.0, 1.0], [0.0, 1.0], num_bins=10) == pytest.approx(0.0)
    # Máxima descalibración: conf=1.0 con acc=0.0 -> 100.0 %
    assert compute_ece([1.0, 1.0], [0.0, 0.0], num_bins=10) == pytest.approx(100.0)
    # 2 bins poblados con 1 muestra c/u: 0.5*|1.0-0.9| + 0.5*|0.0-0.2| = 0.15 -> 15.0 %
    assert compute_ece([0.9, 0.2], [1.0, 0.0], num_bins=10) == pytest.approx(15.0, abs=1e-6)
    # Vacío o longitudes distintas -> 0.0
    assert compute_ece([], []) == 0.0
    assert compute_ece([0.5], [1.0, 0.0]) == 0.0


class _StubCausalModel:
    """Doble determinista para validar desacoplamiento de métricas en evaluate_causal."""

    def __init__(self, responses: dict[bytes, str]):
        self.training = False
        self._responses = responses

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def generate_causal_two_step(self, prompt_b: bytes):
        return self._responses[prompt_b], {"digit_confidences": [1.0] * 12}


def test_evaluate_causal_decouples_template_validity_from_arithmetic_em():
    cfg = HybridConfig(use_scratchpad=True)
    ds = ProceduralCompositionalDataset(cfg, split="dev", num_samples=2)
    p0, t0, _, _ = ds.samples[0]
    p1, t1, _, _ = ds.samples[1]

    # Muestra 0: exacta; Muestra 1: plantilla 100 % válida pero todos los dígitos mal (000)
    wrong_valid = "pw: 000\npms: 000\nwatt: 000\nms: 000\n"
    assert wrong_valid != t1.decode("utf-8")

    stub = _StubCausalModel({p0: t0.decode("utf-8"), p1: wrong_valid})
    metrics = evaluate_causal(stub, ds, torch.device("cpu"))

    assert metrics["fmt_valid_pct"] == pytest.approx(100.0)
    assert metrics["exact_match_pct"] == pytest.approx(50.0)
    assert metrics["scratchpad_exact_pct"] == pytest.approx(50.0)
    assert metrics["final_exact_pct"] == pytest.approx(50.0)
    assert metrics["digit_acc_pct"] < 100.0


def test_bitlinear_quantize_weights_produces_strict_ternary_codes():
    torch.manual_seed(42)
    layer = BitLinear(32, 16)
    _, q_w, scale_w = layer.quantize_weights()

    unique_vals = set(q_w.unique().tolist())
    assert unique_vals.issubset({-1.0, 0.0, 1.0})
    assert float(scale_w.item()) > 0.0


def test_bitlinear_eval_addsub_matches_reference_modes():
    torch.manual_seed(7)
    layer = BitLinear(64, 32).eval()
    x = torch.randn(3, 8, 64)

    with torch.no_grad():
        y_default = layer(x)
        y_int = layer(x, integer_mode=True)
        y_addsub = layer(x, add_sub_mode=True)

    assert float((y_default - y_int).abs().max().item()) < 1e-4
    assert float((y_int - y_addsub).abs().max().item()) < 1e-4


def test_bitlinear_ste_backward_passes_finite_nonzero_gradients():
    torch.manual_seed(11)
    layer = BitLinear(32, 16).train()
    x = torch.randn(2, 4, 32, requires_grad=True)
    loss = layer(x).sum()
    loss.backward()

    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()
    assert layer.weight.grad.abs().sum().item() > 0.0


def test_bitlinear_loaded_without_fp32_weights_rejects_training_forward():
    layer = BitLinear(16, 8)
    layer.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
    layer.use_loaded_bitcos = True
    layer.train()
    with pytest.raises(RuntimeError, match="No se puede entrenar"):
        layer(torch.randn(1, 4, 16))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
