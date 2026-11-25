# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Suite de serialización Paged BitCOS v1.1 para TRACELanguageModel."""

import pytest
import torch

from tests._helpers import make_small_lm_config
from trace.lm.config import TRACELMConfig
from trace.lm.model import TRACELanguageModel
from trace.serialization.paged_bitcos import (
    PagedBitCOSReader,
    PagedBitCOSSerializer,
    count_fp32_weights_in_bitlinear,
)


def _export_and_reload(tmp_path, cfg: TRACELMConfig):
    torch.manual_seed(42)
    model = TRACELanguageModel(cfg).eval()
    bin_path = str(tmp_path / "test_lm.bin")
    summary = PagedBitCOSSerializer.export_paged_model(model=model, output_path=bin_path, format_type="fast")

    fresh_model = TRACELanguageModel(cfg)
    reader = PagedBitCOSReader(bin_path)
    try:
        arch = reader.get_architecture()
        config_read = reader.get_model_config()
        loaded_count = reader.load_into_model(fresh_model)
        breakdown = reader.get_accounting_breakdown()
        page_meta = (reader.page_size, reader.pages_offset, reader.total_pages, reader.total_size)
        has_dup_head = "lm_head.weight" in reader.tensors_map
    finally:
        reader.close()
    fresh_model.eval()
    return model, fresh_model, summary, arch, config_read, loaded_count, breakdown, page_meta, has_dup_head


def test_exported_metadata_preserves_config_without_duplicate_head(tmp_path):
    cfg = make_small_lm_config(vocab_size=4096, max_seq_len=64)
    _, fresh, summary, arch, config_read, loaded_count, _, _, has_dup_head = _export_and_reload(tmp_path, cfg)

    assert summary["architecture"] == "trace-lm-v1"
    assert arch == "trace-lm-v1"
    assert config_read is not None and config_read["d_model"] == cfg.d_model
    assert loaded_count > 0
    assert not has_dup_head, "lm_head.weight no debe serializarse por duplicado estando ligado."
    assert count_fp32_weights_in_bitlinear(fresh) == 0
    assert fresh.lm_head.weight is fresh.tok_emb.weight


@pytest.mark.parametrize(
    "kwargs,tol",
    [
        ({}, 1e-6),
        ({"use_weight_absorption": True}, 1e-6),
        ({"quantize_kv_int8": True}, 1e-6),
        ({"add_sub_mode": True}, 0.0),
    ],
)
def test_reloaded_model_logits_match_original(tmp_path, kwargs: dict, tol: float):
    cfg = make_small_lm_config(vocab_size=4096, max_seq_len=64)
    orig, loaded, *_ = _export_and_reload(tmp_path, cfg)
    input_ids = torch.tensor([[10, 45, 120, 258, 80, 99]], dtype=torch.long)

    with torch.no_grad():
        orig_logits, _, _ = orig(input_ids, **kwargs)
        loaded_logits, _, _ = loaded(input_ids, **kwargs)

    diff = float(torch.max(torch.abs(orig_logits - loaded_logits)).item())
    if tol == 0.0:
        assert diff == 0.0
    else:
        assert diff < tol, f"Discrepancia en {kwargs}: {diff:.6e}"


def test_reloaded_model_greedy_generation_matches_original(tmp_path):
    cfg = make_small_lm_config(vocab_size=4096, max_seq_len=64)
    orig, loaded, *_ = _export_and_reload(tmp_path, cfg)
    input_ids = torch.tensor([[10, 45, 120, 258, 80, 99]], dtype=torch.long)

    with torch.no_grad():
        orig_gen = orig.generate(input_ids, max_new_tokens=8, temperature=0.0)
        loaded_gen = loaded.generate(input_ids, max_new_tokens=8, temperature=0.0)

    assert torch.equal(orig_gen, loaded_gen)


def test_export_audit_25m_preset_alignment_contract(tmp_path):
    cfg = TRACELMConfig.preset_25m()
    _, fresh, summary, _, _, _, breakdown, page_meta, _ = _export_and_reload(tmp_path, cfg)
    page_size, pages_offset, total_pages, total_size = page_meta

    assert summary["architecture"] == "trace-lm-v1"
    assert page_size == 4096
    assert pages_offset % page_size == 0
    assert total_pages * page_size == (total_size - pages_offset)
    assert count_fp32_weights_in_bitlinear(fresh) == 0
    assert fresh.lm_head.weight is fresh.tok_emb.weight
    assert breakdown["resident_fp32_params_bytes"] > 0
    assert 0 < breakdown["ternary_pages_bytes"] < 8 * 1024 * 1024


def test_corrupt_bin_magic_rejected(tmp_path):
    corrupt_path = tmp_path / "corrupt_magic.bin"
    corrupt_path.write_bytes(b"BADMAGIC" + bytes(120))
    with pytest.raises((AssertionError, ValueError), match="Magic"):
        PagedBitCOSReader(str(corrupt_path))


def test_truncated_bin_header_rejected(tmp_path):
    trunc_path = tmp_path / "truncated.bin"
    trunc_path.write_bytes(b"PBITCOS1")
    with pytest.raises(ValueError, match="truncado"):
        PagedBitCOSReader(str(trunc_path))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
