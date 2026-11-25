# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Tests unitarios del harness de perplejidad y puerta ±1% de Fase 4A.4."""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scripts.benchmark_bitnet_2b import (
    check_ppl_exit_gate,
    compute_token_nll,
    evaluate_perplexity_c,
    load_native_libs,
)
from tests.test_gqa_compact_emb import _export_gqa_model


def test_compute_token_nll_matches_pytorch_cross_entropy():
    rng = np.random.default_rng(7)
    logits = (rng.standard_normal(512) * 3.0 + 1000.0).astype(np.float32)
    target = 137
    got = compute_token_nll(logits, target)
    expected = float(
        F.cross_entropy(
            torch.from_numpy(logits).double().unsqueeze(0),
            torch.tensor([target], dtype=torch.long),
        ).item()
    )
    assert abs(got - expected) < 1e-10


def test_evaluate_perplexity_c_matches_pytorch_gqa_oracle(tmp_path):
    lm, _ = load_native_libs()
    cfg, model, bin_path = _export_gqa_model(tmp_path, seed=61, embedding_dtype="float32", max_seq_len=128)
    tokens = [(i * 19 + 3) % cfg.vocab_size for i in range(40)]

    c_model = lm.trace_lm_load(bin_path.encode())
    assert c_model is not None
    try:
        c_res = evaluate_perplexity_c(
            lm, c_model, tokens, context_len=64, chunk_size=12, use_int8_kv=0
        )
    finally:
        lm.trace_lm_free(c_model)

    with torch.no_grad():
        py_logits, _ = model(torch.tensor([tokens[:-1]], dtype=torch.long), quantize_kv_int8=False)
        py_nll = float(
            F.cross_entropy(
                py_logits.squeeze(0).double(),
                torch.tensor(tokens[1:], dtype=torch.long),
            ).item()
        )
    py_ppl = math.exp(py_nll)
    assert c_res["tokens_evaluated"] == len(tokens) - 1
    rel_err = abs(c_res["perplexity"] - py_ppl) / py_ppl
    assert rel_err < 1e-4, f"PPL C ({c_res['perplexity']:.6f}) vs PyTorch ({py_ppl:.6f}): rel={rel_err:.6e}"


def test_check_ppl_exit_gate_enforces_one_percent_tolerance():
    ok = check_ppl_exit_gate(ppl=10.09, ref_ppl=10.0, max_rel_diff=0.01)
    assert ok["passed"] is True
    bad = check_ppl_exit_gate(ppl=10.12, ref_ppl=10.0, max_rel_diff=0.01)
    assert bad["passed"] is False
    none_ref = check_ppl_exit_gate(ppl=10.0, ref_ppl=None)
    assert none_ref["passed"] is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
