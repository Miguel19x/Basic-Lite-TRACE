# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Tests unitarios de Paged BitCOS v1 (round-trip, 0 FP32 en RAM, bordes y paridad forward)."""

import numpy as np
import pytest
import torch

from config import HybridConfig
from trace.models.causal import TRACECausalModel
from trace.ops.paged_kernel import unpack_page_to_dense
from trace.serialization.paged_bitcos import (
    PagedBitCOSReader,
    PagedBitCOSSerializer,
    count_fp32_weights_in_bitlinear,
)


@pytest.mark.parametrize("format_type", ["compact", "fast"])
@pytest.mark.parametrize("cols", [1, 7, 31, 32, 33, 50, 100, 192, 384])
def test_pack_unpack_roundtrip_preserves_weights(format_type: str, cols: int):
    rng = np.random.RandomState(42 + cols)
    rows = 16
    expected_scale = 0.05
    raw = rng.randint(-1, 2, size=(rows, cols)).astype(np.int8)

    page_bytes = PagedBitCOSSerializer.pack_ternary_page(
        row_start=0,
        out_features=rows,
        in_features=cols,
        scale_w=expected_scale,
        q_w_chunk=raw,
        page_size=4096,
        format_type=format_type,
    )
    r_s, r_cnt, c_cnt, scale_w, codes = unpack_page_to_dense(page_bytes)

    assert (r_s, r_cnt, c_cnt) == (0, rows, cols)
    assert abs(scale_w - expected_scale) < 1e-7
    assert np.array_equal(codes.numpy(), raw)


@pytest.mark.parametrize("format_type", ["compact", "fast"])
def test_loaded_model_has_zero_fp32_bitlinear_weights(tmp_path, format_type: str):
    torch.manual_seed(42)
    cfg = HybridConfig(device="cpu", num_recursions=1, adaptive_recursion=False)
    model = TRACECausalModel(cfg).eval()

    bin_path = str(tmp_path / f"paged_{format_type}.bin")
    PagedBitCOSSerializer.export_paged_model(model, bin_path, format_type=format_type)

    fresh_model = TRACECausalModel(cfg)
    reader = PagedBitCOSReader(bin_path)
    try:
        reader.load_into_model(fresh_model)
    finally:
        reader.close()

    assert count_fp32_weights_in_bitlinear(fresh_model) == 0


def test_read_page_bytes_rejects_out_of_range_ids(tmp_path):
    torch.manual_seed(42)
    cfg = HybridConfig(device="cpu", num_recursions=1, adaptive_recursion=False)
    model = TRACECausalModel(cfg).eval()
    bin_path = str(tmp_path / "bounds.bin")
    PagedBitCOSSerializer.export_paged_model(model, bin_path, format_type="fast")

    reader = PagedBitCOSReader(bin_path)
    try:
        for bad_id in (-1, reader.total_pages, reader.total_pages + 10):
            with pytest.raises(IndexError):
                reader.read_page_bytes(bad_id)
    finally:
        reader.close()


@pytest.mark.parametrize("format_type", ["compact", "fast"])
@pytest.mark.parametrize("use_add_sub", [False, True])
def test_forward_pass_loaded_bitcos_matches_reference_with_zero_fp32(
    tmp_path, format_type: str, use_add_sub: bool
):
    torch.manual_seed(123)
    cfg = HybridConfig(
        device="cpu",
        num_recursions=1,
        adaptive_recursion=False,
        use_add_sub_inference=use_add_sub,
    )
    model = TRACECausalModel(cfg).eval()
    x = torch.randint(0, 256, (1, 32), dtype=torch.long)
    t_noise = torch.ones(1)

    with torch.no_grad():
        orig_logits = model.forward_backbone(x, t_noise, stage_id=0)

    bin_path = str(tmp_path / f"fwd_{format_type}_{int(use_add_sub)}.bin")
    PagedBitCOSSerializer.export_paged_model(model, bin_path, format_type=format_type)

    fresh_model = TRACECausalModel(cfg)
    reader = PagedBitCOSReader(bin_path)
    try:
        reader.load_into_model(fresh_model)
    finally:
        reader.close()
    fresh_model.eval()

    with torch.no_grad():
        loaded_logits = fresh_model.forward_backbone(x, t_noise, stage_id=0)

    assert count_fp32_weights_in_bitlinear(fresh_model) == 0
    diff = float((orig_logits - loaded_logits).abs().max().item())
    assert diff < 1e-4, f"Paridad fallida (format={format_type}, add_sub={use_add_sub}): {diff:.6e}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
