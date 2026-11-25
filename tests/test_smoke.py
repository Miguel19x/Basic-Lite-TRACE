# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Tests de humo para carga del kernel C y contrato de salida de TRACECausalModel."""

import pytest
import torch

from config import HybridConfig
from data import ProceduralCompositionalDataset
from tests._helpers import load_paged_gemm_lib
from trace.models.causal import TRACECausalModel


def test_kernel_c_library_exposes_required_symbols():
    lib = load_paged_gemm_lib()
    required_symbols = (
        "paged_direct_matmul_fast_c",
        "paged_direct_matmul_fast_c_multi",
        "paged_lowrank_matmul_row_c",
        "ternary_addsub_cpu_kernel",
        "paged_absorb_uk_f32",
        "paged_absorb_uv_f32",
        "trace_dot_f32",
        "trace_axpy_f32",
        "trace_cpu_active_isa",
        "trace_cpu_set_isa",
    )
    for sym in required_symbols:
        assert hasattr(lib, sym), f"Símbolo '{sym}' ausente en libpaged_gemm.so."


def test_kernel_c_reports_valid_active_isa():
    lib = load_paged_gemm_lib()
    active_isa = lib.trace_cpu_active_isa().decode("utf-8")
    assert active_isa in ("scalar", "avx2"), f"ISA inesperada: '{active_isa}'"


def test_trace_causal_model_forward_produces_valid_fixed_width_output():
    torch.manual_seed(42)
    cfg = HybridConfig(device="cpu", num_recursions=1, adaptive_recursion=False)
    model = TRACECausalModel(cfg).eval()

    dataset = ProceduralCompositionalDataset(cfg, split="dev")
    prompt_b, _, _, _ = dataset.samples[0]

    seq = [cfg.bos_token_id] + list(prompt_b) + [cfg.eos_token_id]
    seq = seq + [cfg.pad_token_id] * max(0, cfg.max_seq_len - len(seq))
    x = torch.tensor([seq[: cfg.max_seq_len]], dtype=torch.long)

    with torch.no_grad():
        logits = model.forward_backbone(x, torch.ones(1), stage_id=0)
        gen_str, meta = model.generate_causal_two_step(prompt_b)

    assert logits.shape[0] == 1 and logits.shape[2] == cfg.vocab_size
    assert torch.isfinite(logits).all(), "Logits de forward_backbone contienen NaN/Inf."

    assert len(gen_str) == 35, f"Longitud de salida fija inesperada: {len(gen_str)} != 35"
    assert gen_str[0:4] == "pw: "
    assert gen_str[7:13] == "\npms: "
    assert gen_str[16:23] == "\nwatt: "
    assert gen_str[26:31] == "\nms: "
    assert gen_str[34:35] == "\n"
    assert "watt" in meta and "ms" in meta
    assert len(meta["digit_confidences"]) == 12
    assert all(0.0 <= c <= 1.0 for c in meta["digit_confidences"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
