# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Suite de paridad numérica y falsación activa de Fase 3 (libtrace_lm.so vs PyTorch)."""

import ctypes
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests._helpers import (
    PAGED_HEADER_SIZE,
    flip_ternary_payload_inplace,
    load_trace_lm_lib,
    make_small_lm_config,
)
from trace.lm.model import TRACELanguageModel, TRACELMConfig
from trace.serialization.paged_bitcos import PagedBitCOSReader, PagedBitCOSSerializer


def _run_c_lm_logits(lib, bin_path: str, prompt: list[int], vocab_size: int, use_int8_kv: int = 1) -> np.ndarray:
    c_model = lib.trace_lm_load(bin_path.encode())
    assert c_model is not None
    c_ws = c_kv = None
    try:
        c_cfg = lib.trace_lm_get_config(c_model)
        c_ws = lib.trace_lm_workspace_create(c_cfg, max(128, len(prompt)))
        c_kv = lib.trace_lm_kv_create(c_cfg, 256, use_int8_kv)
        assert c_ws is not None and c_kv is not None

        c_logits = (ctypes.c_float * (len(prompt) * vocab_size))()
        c_tokens = (ctypes.c_int * len(prompt))(*prompt)
        assert lib.trace_lm_forward(c_model, c_kv, c_tokens, len(prompt), c_logits, 0, c_ws) == len(prompt)
        return np.ctypeslib.as_array(c_logits).reshape(len(prompt), vocab_size).copy()
    finally:
        if c_kv is not None:
            lib.trace_lm_kv_free(c_kv)
        if c_ws is not None:
            lib.trace_lm_workspace_free(c_ws)
        lib.trace_lm_free(c_model)


def test_strict_parity_2layer_int8_kv(tmp_path):
    torch.manual_seed(42)
    lib = load_trace_lm_lib()
    cfg = make_small_lm_config(vocab_size=4096, max_seq_len=256)
    model = TRACELanguageModel(cfg).eval()
    prompt = [15, 42, 88, 120, 200, 50, 12, 99]
    input_ids = torch.tensor([prompt], dtype=torch.long)

    bin_path = str(tmp_path / "strict_2l.bin")
    PagedBitCOSSerializer.export_paged_model(model=model, output_path=bin_path, format_type="fast")

    arr_c = _run_c_lm_logits(lib, bin_path, prompt, cfg.vocab_size, use_int8_kv=1)
    with torch.no_grad():
        py_logits, _, _ = model(input_ids, use_weight_absorption=True, quantize_kv_int8=True, add_sub_mode=True)
    arr_py = py_logits.squeeze(0).numpy()

    assert float(np.max(np.abs(arr_c - arr_py))) <= 1e-4
    assert np.array_equal(np.argmax(arr_c, axis=-1), np.argmax(arr_py, axis=-1))


def test_strict_parity_2layer_vs_fp32_ppl(tmp_path):
    torch.manual_seed(42)
    lib = load_trace_lm_lib()
    cfg = make_small_lm_config(vocab_size=4096, max_seq_len=256)
    model = TRACELanguageModel(cfg).eval()
    prompt = [15, 42, 88, 120, 200, 50, 12, 99]
    input_ids = torch.tensor([prompt], dtype=torch.long)
    targets = input_ids[:, 1:].contiguous()

    bin_path = str(tmp_path / "strict_2l_ppl.bin")
    PagedBitCOSSerializer.export_paged_model(model=model, output_path=bin_path, format_type="fast")

    arr_c = _run_c_lm_logits(lib, bin_path, prompt, cfg.vocab_size, use_int8_kv=1)
    with torch.no_grad():
        py_fp32, _, _ = model(input_ids, use_weight_absorption=True, quantize_kv_int8=False, add_sub_mode=True)
    arr_py = py_fp32.squeeze(0).numpy()

    loss_c = F.cross_entropy(torch.from_numpy(arr_c[:-1]), targets[0]).item()
    loss_py = F.cross_entropy(torch.from_numpy(arr_py[:-1]), targets[0]).item()
    rel_delta_ppl_pct = (abs(math.exp(loss_c) - math.exp(loss_py)) / math.exp(loss_py)) * 100.0
    greedy_match = float(np.mean(np.argmax(arr_c, axis=-1) == np.argmax(arr_py, axis=-1)) * 100.0)

    assert abs(loss_c - loss_py) < 0.05
    assert rel_delta_ppl_pct < 1.0
    assert greedy_match >= 95.0


@pytest.mark.parametrize(
    "use_int8_kv,max_diff_tol,min_greedy_pct",
    [
        (1, 0.08, 90.0),
        (0, 0.05, 95.0),
    ],
)
def test_scale_parity_preset_25m(tmp_path, use_int8_kv: int, max_diff_tol: float, min_greedy_pct: float):
    torch.manual_seed(42)
    lib = load_trace_lm_lib()
    cfg = TRACELMConfig.preset_25m()
    cfg.max_seq_len = 256
    model = TRACELanguageModel(cfg).eval()

    prompt = [(i * 37 + 11) % cfg.vocab_size for i in range(64)]
    input_ids = torch.tensor([prompt], dtype=torch.long)
    targets = input_ids[:, 1:].contiguous()

    bin_path = str(tmp_path / f"model_25m_kv{use_int8_kv}.bin")
    PagedBitCOSSerializer.export_paged_model(model=model, output_path=bin_path, format_type="fast")

    arr_c = _run_c_lm_logits(lib, bin_path, prompt, cfg.vocab_size, use_int8_kv=use_int8_kv)
    with torch.no_grad():
        py_logits, _, _ = model(
            input_ids,
            use_weight_absorption=True,
            quantize_kv_int8=(use_int8_kv == 1),
            add_sub_mode=True,
        )
    arr_py = py_logits.squeeze(0).numpy()

    diff = float(np.max(np.abs(arr_c - arr_py)))
    greedy_match = float(np.mean(np.argmax(arr_c, axis=-1) == np.argmax(arr_py, axis=-1)) * 100.0)
    loss_c = F.cross_entropy(torch.from_numpy(arr_c[:-1]), targets[0]).item()
    loss_py = F.cross_entropy(torch.from_numpy(arr_py[:-1]), targets[0]).item()
    rel_delta_ppl = (abs(math.exp(loss_c) - math.exp(loss_py)) / math.exp(loss_py)) * 100.0

    assert diff < max_diff_tol
    assert greedy_match >= min_greedy_pct
    assert rel_delta_ppl < 0.5


@pytest.mark.parametrize("tensor_key", ["layers.0.attn.w_dkv", "layers.0.attn.w_kr"])
def test_falsification_weight_sign_flip_degrades_logits(tmp_path, tensor_key: str):
    torch.manual_seed(1337)
    lib = load_trace_lm_lib()
    cfg = make_small_lm_config(vocab_size=4096, max_seq_len=256)
    model = TRACELanguageModel(cfg).eval()
    prompt = [15, 42, 88, 120, 200, 50, 12, 99, 101, 202, 303, 404, 505, 606, 707, 808]

    canonical_bin = str(tmp_path / "canonical.bin")
    PagedBitCOSSerializer.export_paged_model(model=model, output_path=canonical_bin, format_type="fast")
    arr_sano = _run_c_lm_logits(lib, canonical_bin, prompt, cfg.vocab_size, use_int8_kv=1)

    reader = PagedBitCOSReader(canonical_bin)
    try:
        pgs_offset = reader.pages_offset
        page_size = reader.page_size
        page_id = reader.tensors_map[tensor_key]["page_ids"][0]
    finally:
        reader.close()

    with open(canonical_bin, "rb") as f:
        mutated = bytearray(f.read())
    page_start = pgs_offset + page_id * page_size
    flip_ternary_payload_inplace(mutated, page_start + PAGED_HEADER_SIZE, page_start + page_size)

    mut_bin = tmp_path / f"mut_{tensor_key.replace('.', '_')}.bin"
    mut_bin.write_bytes(mutated)
    arr_mut = _run_c_lm_logits(lib, str(mut_bin), prompt, cfg.vocab_size, use_int8_kv=1)
    assert float(np.max(np.abs(arr_mut - arr_sano))) > 0.1


def test_falsification_prefix_token_mutation_invalidates_cache(tmp_path):
    torch.manual_seed(1337)
    lib = load_trace_lm_lib()
    cfg = make_small_lm_config(vocab_size=4096, max_seq_len=256)
    model = TRACELanguageModel(cfg).eval()
    prompt_a = [15, 42, 88, 120, 200, 50, 12, 99, 101, 202, 303, 404, 505, 606, 707, 808]
    prompt_b = list(prompt_a)
    prompt_b[10] = (prompt_b[10] + 777) % cfg.vocab_size

    bin_path = str(tmp_path / "pfx_falsify.bin")
    PagedBitCOSSerializer.export_paged_model(model=model, output_path=bin_path, format_type="fast")

    c_model = lib.trace_lm_load(bin_path.encode())
    assert c_model is not None
    c_ws = c_kv = c_kv_clean = None
    try:
        c_cfg = lib.trace_lm_get_config(c_model)
        c_ws = lib.trace_lm_workspace_create(c_cfg, 128)
        c_kv = lib.trace_lm_kv_create(c_cfg, 256, 1)
        c_kv_clean = lib.trace_lm_kv_create(c_cfg, 256, 1)
        assert c_ws is not None and c_kv is not None and c_kv_clean is not None

        logits_a = (ctypes.c_float * cfg.vocab_size)()
        c_pa = (ctypes.c_int * len(prompt_a))(*prompt_a)
        assert lib.trace_lm_forward(c_model, c_kv, c_pa, len(prompt_a), logits_a, 1, c_ws) == len(prompt_a)
        arr_a = np.ctypeslib.as_array(logits_a).copy()

        logits_b_cached = (ctypes.c_float * cfg.vocab_size)()
        c_pb = (ctypes.c_int * len(prompt_b))(*prompt_b)
        s_proc = lib.trace_lm_forward(c_model, c_kv, c_pb, len(prompt_b), logits_b_cached, 1, c_ws)
        assert lib.trace_lm_kv_get_last_common_prefix(c_kv) == 10
        assert s_proc == len(prompt_b) - 10

        arr_b_cached = np.ctypeslib.as_array(logits_b_cached)
        assert float(np.max(np.abs(arr_b_cached - arr_a))) > 0.1

        logits_b_clean = (ctypes.c_float * cfg.vocab_size)()
        assert lib.trace_lm_forward(c_model, c_kv_clean, c_pb, len(prompt_b), logits_b_clean, 1, c_ws) == len(prompt_b)
        assert float(np.max(np.abs(arr_b_cached - np.ctypeslib.as_array(logits_b_clean)))) < 1e-5
    finally:
        if c_kv is not None:
            lib.trace_lm_kv_free(c_kv)
        if c_kv_clean is not None:
            lib.trace_lm_kv_free(c_kv_clean)
        if c_ws is not None:
            lib.trace_lm_workspace_free(c_ws)
        lib.trace_lm_free(c_model)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
