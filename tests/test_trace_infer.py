# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Suite de validación del motor nativo C causal (libtrace_infer.so)."""

import copy
import ctypes

import numpy as np
import pytest
import torch

from config import HybridConfig
from data import ProceduralCompositionalDataset
from tests._helpers import (
    REAL_CAUSAL_BIN,
    REAL_CAUSAL_CKPT,
    has_real_causal_checkpoints,
    load_trace_infer_lib,
)
from trace.models.causal import TRACECausalModel
from trace.modules.attention import build_target_unified_mask
from trace.serialization.paged_bitcos import PagedBitCOSSerializer


def _build_synthetic_causal_bin(tmp_path):
    torch.manual_seed(42)
    cfg = HybridConfig(device="cpu", cpu_threads=2, use_add_sub_inference=True)
    model_ref = TRACECausalModel(cfg).eval()
    bin_path = str(tmp_path / "synthetic_causal.bin")
    PagedBitCOSSerializer.export_paged_model(model_ref, bin_path, format_type="fast")
    return cfg, model_ref, bin_path


def _prepare_sample_sequence(cfg: HybridConfig):
    dataset_dev = ProceduralCompositionalDataset(cfg, split="dev")
    prompt_b = dataset_dev.samples[0][0]
    template_target = b"pw: ???\npms: ???\nwatt: ???\nms: ???\n"
    seq = [cfg.bos_token_id] + list(prompt_b) + list(template_target) + [cfg.eos_token_id]
    full_seq = seq + [cfg.pad_token_id] * max(0, cfg.max_seq_len - len(seq))
    return prompt_b, 1 + len(prompt_b), torch.tensor([full_seq], dtype=torch.long), (ctypes.c_int32 * cfg.max_seq_len)(*full_seq)


def test_c_causal_loader_aligns_pages_to_4kb(tmp_path):
    lib = load_trace_infer_lib()
    _, _, bin_path = _build_synthetic_causal_bin(tmp_path)
    model_buf = ctypes.create_string_buffer(lib.trace_model_sizeof())

    assert lib.trace_model_load_odirect(bin_path.encode("utf-8"), model_buf) == 0
    try:
        assert lib.trace_model_get_pages_offset(model_buf) % 4096 == 0
        assert lib.trace_model_get_pread_calls(model_buf) >= 2
    finally:
        lib.trace_model_free(model_buf)


@pytest.mark.parametrize("stage_id", [0, 1, 2, 3])
def test_c_causal_backbone_matches_pytorch_per_stage(tmp_path, stage_id: int):
    lib = load_trace_infer_lib()
    cfg, model_ref, bin_path = _build_synthetic_causal_bin(tmp_path)
    _, _, x_tensor, c_token_ids = _prepare_sample_sequence(cfg)

    model_buf = ctypes.create_string_buffer(lib.trace_model_sizeof())
    assert lib.trace_model_load_odirect(bin_path.encode("utf-8"), model_buf) == 0
    try:
        with torch.no_grad():
            ref_logits = model_ref.forward_backbone(x_tensor, torch.ones(1), stage_id=stage_id).numpy()[0]
        L_trim = ref_logits.shape[0]
        c_logits_buf = (ctypes.c_float * (L_trim * cfg.vocab_size))()
        lib.trace_forward_backbone(model_buf, c_token_ids, L_trim, stage_id, 0, c_logits_buf)
        c_logits_arr = np.ctypeslib.as_array(c_logits_buf).reshape(L_trim, cfg.vocab_size)

        diff = float(np.max(np.abs(ref_logits - c_logits_arr)))
        assert diff < 0.1, f"Discrepancia etapa {stage_id}: {diff:.6e}"
    finally:
        lib.trace_model_free(model_buf)


def test_c_causal_unified_mask_matches_pytorch(tmp_path):
    lib = load_trace_infer_lib()
    cfg, model_ref, bin_path = _build_synthetic_causal_bin(tmp_path)
    _, plen, x_tensor, c_token_ids = _prepare_sample_sequence(cfg)
    attn_mask = build_target_unified_mask([plen], cfg.max_seq_len, cfg.block_size, torch.device("cpu"))

    model_buf = ctypes.create_string_buffer(lib.trace_model_sizeof())
    assert lib.trace_model_load_odirect(bin_path.encode("utf-8"), model_buf) == 0
    try:
        with torch.no_grad():
            ref_masked = model_ref.forward_backbone(x_tensor, torch.ones(1), stage_id=1, attn_mask=attn_mask).numpy()[0]
        L_masked = ref_masked.shape[0]
        c_masked_buf = (ctypes.c_float * (L_masked * cfg.vocab_size))()
        lib.trace_forward_backbone(model_buf, c_token_ids, L_masked, 1, plen, c_masked_buf)
        c_masked_arr = np.ctypeslib.as_array(c_masked_buf).reshape(L_masked, cfg.vocab_size)
        assert float(np.max(np.abs(ref_masked - c_masked_arr))) < 0.1
    finally:
        lib.trace_model_free(model_buf)


def test_c_causal_lazy_backbone_matches_full_backbone(tmp_path):
    lib = load_trace_infer_lib()
    cfg, _, bin_path = _build_synthetic_causal_bin(tmp_path)
    _, plen, _, c_token_ids = _prepare_sample_sequence(cfg)
    L_masked = cfg.max_seq_len
    target_indices = [plen, plen + 1, min(plen + 4, L_masked - 1)]

    model_buf = ctypes.create_string_buffer(lib.trace_model_sizeof())
    assert lib.trace_model_load_odirect(bin_path.encode("utf-8"), model_buf) == 0
    try:
        c_full_buf = (ctypes.c_float * (L_masked * cfg.vocab_size))()
        lib.trace_forward_backbone(model_buf, c_token_ids, L_masked, 1, plen, c_full_buf)
        full_arr = np.ctypeslib.as_array(c_full_buf).reshape(L_masked, cfg.vocab_size)

        c_targets = (ctypes.c_int * len(target_indices))(*target_indices)
        c_lazy_buf = (ctypes.c_float * (len(target_indices) * cfg.vocab_size))()
        lib.trace_forward_backbone_lazy(
            model_buf, c_token_ids, L_masked, 1, plen, c_targets, len(target_indices), c_lazy_buf
        )
        lazy_arr = np.ctypeslib.as_array(c_lazy_buf).reshape(len(target_indices), cfg.vocab_size)
        assert float(np.max(np.abs(lazy_arr - full_arr[target_indices, :]))) <= 1e-6
    finally:
        lib.trace_model_free(model_buf)


def test_c_causal_two_step_generation_keeps_pread_constant(tmp_path):
    lib = load_trace_infer_lib()
    cfg, model_ref, bin_path = _build_synthetic_causal_bin(tmp_path)
    model_buf = ctypes.create_string_buffer(lib.trace_model_sizeof())
    assert lib.trace_model_load_odirect(bin_path.encode("utf-8"), model_buf) == 0

    try:
        pread_cold = lib.trace_model_get_pread_calls(model_buf)
        for split_name in ("dev", "test-composicional"):
            dataset = ProceduralCompositionalDataset(cfg, split=split_name)
            for i in range(2):
                prompt_b = dataset.samples[i][0]
                out_c_buf = ctypes.create_string_buffer(512)
                assert lib.trace_generate_causal_two_step(model_buf, prompt_b, len(prompt_b), out_c_buf, 512) >= 0
                with torch.no_grad():
                    ref_str, _ = model_ref.generate_causal_two_step(prompt_b)
                c_bytes = out_c_buf.value
                # Normalizar bytes no-ASCII igual que TRACECausalModel.generate_causal_two_step
                c_norm = "".join(chr(b) if (32 <= b < 127 or b == 10) else "?" for b in c_bytes)
                assert c_norm == ref_str
                assert lib.trace_model_get_pread_calls(model_buf) == pread_cold
    finally:
        lib.trace_model_free(model_buf)


@pytest.mark.parametrize("kind", ["missing", "corrupt"])
def test_c_causal_loader_rejects_invalid_binary(tmp_path, kind: str):
    lib = load_trace_infer_lib()
    bad_path = tmp_path / f"{kind}.bin"
    if kind == "corrupt":
        bad_path.write_bytes(b"BAD_CAUSAL_MAGIC_HEADER_0000")
    model_buf = ctypes.create_string_buffer(lib.trace_model_sizeof())
    assert lib.trace_model_load_odirect(str(bad_path).encode("utf-8"), model_buf) != 0


@pytest.mark.skipif(not has_real_causal_checkpoints(), reason="Checkpoints locales Fase 1 ausentes.")
def test_c_causal_real_checkpoint_text_parity():
    lib = load_trace_infer_lib()
    ckpt = torch.load(REAL_CAUSAL_CKPT, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(ckpt["cfg"])
    cfg.device = "cpu"
    cfg.cpu_threads = 2
    cfg.use_add_sub_inference = True

    model_ref = TRACECausalModel(cfg)
    model_ref.load_state_dict(ckpt["model_state"], strict=True)
    model_ref.eval()

    model_buf = ctypes.create_string_buffer(lib.trace_model_sizeof())
    assert lib.trace_model_load_odirect(REAL_CAUSAL_BIN.encode("utf-8"), model_buf) == 0
    try:
        pread_cold = lib.trace_model_get_pread_calls(model_buf)
        for split_name in ("dev", "test-composicional"):
            dataset = ProceduralCompositionalDataset(cfg, split=split_name)
            for i in range(4):
                prompt_b = dataset.samples[i][0]
                out_c_buf = ctypes.create_string_buffer(512)
                assert lib.trace_generate_causal_two_step(model_buf, prompt_b, len(prompt_b), out_c_buf, 512) >= 0
                with torch.no_grad():
                    ref_str, _ = model_ref.generate_causal_two_step(prompt_b)
                c_bytes = out_c_buf.value
                # Normalizar bytes no-ASCII igual que TRACECausalModel.generate_causal_two_step
                c_norm = "".join(chr(b) if (32 <= b < 127 or b == 10) else "?" for b in c_bytes)
                assert c_norm == ref_str
                assert lib.trace_model_get_pread_calls(model_buf) == pread_cold
    finally:
        lib.trace_model_free(model_buf)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
