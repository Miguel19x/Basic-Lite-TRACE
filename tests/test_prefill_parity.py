# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Tests de caracterización y paridad estricta para cold prefill (S > 1) en libtrace_lm.so."""

import ctypes
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from tests._helpers import load_trace_lm_lib
from trace.lm.config import TRACELMConfig
from trace.lm.model import TRACELanguageModel
from trace.serialization.paged_bitcos import PagedBitCOSSerializer


def _make_prefill_config(max_seq_len: int = 512) -> TRACELMConfig:
    return TRACELMConfig(
        vocab_size=512,
        max_seq_len=max_seq_len,
        d_model=256,
        n_layers=2,
        n_heads=8,
        d_ff=512,
        kv_lora_rank=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        tie_word_embeddings=True,
    )


def _export_prefill_model(tmp_path, seed: int = 42, max_seq_len: int = 512):
    torch.manual_seed(seed)
    cfg = _make_prefill_config(max_seq_len=max_seq_len)
    model = TRACELanguageModel(cfg).eval()
    bin_path = str(tmp_path / f"prefill_s{seed}.bin")
    PagedBitCOSSerializer.export_paged_model(model=model, output_path=bin_path, format_type="fast")
    return cfg, model, bin_path


@pytest.mark.parametrize("use_int8_kv", [0, 1])
def test_c_lm_long_prefill_matches_pytorch(tmp_path, use_int8_kv: int):
    lib = load_trace_lm_lib()
    cfg, model, bin_path = _export_prefill_model(tmp_path, seed=42, max_seq_len=512)
    seq_len = 256
    prompt = [(i * 29 + 7) % cfg.vocab_size for i in range(seq_len)]
    input_ids = torch.tensor([prompt], dtype=torch.long)

    c_model = lib.trace_lm_load(bin_path.encode())
    assert c_model is not None
    c_ws = c_kv = None
    try:
        c_cfg = lib.trace_lm_get_config(c_model)
        c_ws = lib.trace_lm_workspace_create(c_cfg, seq_len)
        c_kv = lib.trace_lm_kv_create(c_cfg, 512, use_int8_kv)
        assert c_ws is not None and c_kv is not None

        c_tokens = (ctypes.c_int * seq_len)(*prompt)
        c_logits = (ctypes.c_float * cfg.vocab_size)()
        assert lib.trace_lm_forward(c_model, c_kv, c_tokens, seq_len, c_logits, 1, c_ws) == seq_len

        with torch.no_grad():
            py_logits, _, _ = model(
                input_ids,
                use_weight_absorption=True,
                quantize_kv_int8=(use_int8_kv == 1),
                add_sub_mode=True,
            )

        c_arr = np.ctypeslib.as_array(c_logits)
        py_arr = py_logits[0, -1, :].numpy()
        diff_arr = np.abs(c_arr - py_arr)
        max_diff = float(np.max(diff_arr))
        mean_diff = float(np.mean(diff_arr))
        # Un salto de 1 ULP en la frontera .5 de cuantización INT8 (1/127 ≈ 7.87e-3) entre
        # PyTorch BLAS (FMA) y C99 (-O1 -ffp-contract=off) puede propagar ~1.6e-3 de deriva media
        # y ~6e-3 de pico al atravesar capa 2 + lm_head; exigimos argmax idéntico y cotas acotadas.
        max_tol = 2.5e-2 if use_int8_kv else 1e-2
        mean_tol = 5e-3 if use_int8_kv else 2.5e-3
        assert max_diff < max_tol, f"Divergencia prefill S=256 (int8={use_int8_kv}): max={max_diff:.6e}"
        assert mean_diff < mean_tol, f"Deriva media prefill S=256 (int8={use_int8_kv}): mean={mean_diff:.6e}"
        assert int(np.argmax(c_arr)) == int(np.argmax(py_arr))
    finally:
        if c_kv is not None:
            lib.trace_lm_kv_free(c_kv)
        if c_ws is not None:
            lib.trace_lm_workspace_free(c_ws)
        lib.trace_lm_free(c_model)


def test_c_lm_prefill_is_bit_identical_across_thread_counts(tmp_path):
    cfg, _, bin_path = _export_prefill_model(tmp_path, seed=77, max_seq_len=256)
    out1 = str(tmp_path / "logits_t1.npy")
    out2 = str(tmp_path / "logits_t2.npy")

    snippet = (
        "import ctypes, sys, numpy as np\n"
        "from tests._helpers import load_trace_lm_lib\n"
        "bin_path, out_path, vocab = sys.argv[1], sys.argv[2], int(sys.argv[3])\n"
        "lib = load_trace_lm_lib()\n"
        "m = lib.trace_lm_load(bin_path.encode())\n"
        "c_cfg = lib.trace_lm_get_config(m)\n"
        "ws = lib.trace_lm_workspace_create(c_cfg, 128)\n"
        "kv = lib.trace_lm_kv_create(c_cfg, 256, 1)\n"
        "prompt = [(i * 19 + 5) % vocab for i in range(128)]\n"
        "c_toks = (ctypes.c_int * 128)(*prompt)\n"
        "c_out = (ctypes.c_float * vocab)()\n"
        "assert lib.trace_lm_forward(m, kv, c_toks, 128, c_out, 1, ws) == 128\n"
        "np.save(out_path, np.ctypeslib.as_array(c_out).copy())\n"
        "lib.trace_lm_kv_free(kv); lib.trace_lm_workspace_free(ws); lib.trace_lm_free(m)\n"
    )

    for threads, out_file in (("1", out1), ("2", out2)):
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = threads
        env["PYTHONPATH"] = "."
        res = subprocess.run(
            [sys.executable, "-c", snippet, bin_path, out_file, str(cfg.vocab_size)],
            env=env,
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, res.stderr

    arr1 = np.load(out1)
    arr2 = np.load(out2)
    assert np.array_equal(arr1, arr2), f"No determinista entre 1 y 2 hilos: max_diff={np.max(np.abs(arr1 - arr2))}"


@pytest.mark.parametrize("use_int8_kv", [0, 1])
def test_c_lm_chunked_prefill_matches_single_pass(tmp_path, use_int8_kv: int):
    lib = load_trace_lm_lib()
    cfg, _, bin_path = _export_prefill_model(tmp_path, seed=99, max_seq_len=256)
    seq_len = 192
    prompt = [(i * 31 + 13) % cfg.vocab_size for i in range(seq_len)]

    c_model = lib.trace_lm_load(bin_path.encode())
    assert c_model is not None
    c_ws = c_kv_full = c_kv_chunk = None
    try:
        c_cfg = lib.trace_lm_get_config(c_model)
        c_ws = lib.trace_lm_workspace_create(c_cfg, seq_len)
        c_kv_full = lib.trace_lm_kv_create(c_cfg, 256, use_int8_kv)
        c_kv_chunk = lib.trace_lm_kv_create(c_cfg, 256, use_int8_kv)

        c_tokens = (ctypes.c_int * seq_len)(*prompt)
        logits_full = (ctypes.c_float * cfg.vocab_size)()
        assert lib.trace_lm_forward(c_model, c_kv_full, c_tokens, seq_len, logits_full, 1, c_ws) == seq_len

        logits_chunk = (ctypes.c_float * cfg.vocab_size)()
        for end in (64, 128, 192):
            sub_toks = (ctypes.c_int * end)(*prompt[:end])
            assert lib.trace_lm_forward(c_model, c_kv_chunk, sub_toks, end, logits_chunk, 1, c_ws) == 64

        diff = float(np.max(np.abs(np.ctypeslib.as_array(logits_full) - np.ctypeslib.as_array(logits_chunk))))
        assert diff < 1e-5, f"Prefill por bloques difiere del de una pasada (int8={use_int8_kv}): {diff:.6e}"
    finally:
        if c_kv_full is not None:
            lib.trace_lm_kv_free(c_kv_full)
        if c_kv_chunk is not None:
            lib.trace_lm_kv_free(c_kv_chunk)
        if c_ws is not None:
            lib.trace_lm_workspace_free(c_ws)
        lib.trace_lm_free(c_model)


def test_c_lm_scalar_matches_avx2_long_prefill(tmp_path):
    lib = load_trace_lm_lib()
    cfg, _, bin_path = _export_prefill_model(tmp_path, seed=123, max_seq_len=256)
    seq_len = 64
    prompt = [(i * 17 + 3) % cfg.vocab_size for i in range(seq_len)]

    c_model = lib.trace_lm_load(bin_path.encode())
    assert c_model is not None
    c_ws = c_kv = None
    try:
        c_cfg = lib.trace_lm_get_config(c_model)
        c_ws = lib.trace_lm_workspace_create(c_cfg, seq_len)
        c_kv = lib.trace_lm_kv_create(c_cfg, 256, 1)
        c_tokens = (ctypes.c_int * seq_len)(*prompt)
        logits_scalar = (ctypes.c_float * cfg.vocab_size)()
        logits_avx2 = (ctypes.c_float * cfg.vocab_size)()

        try:
            lib.trace_cpu_set_isa(b"scalar")
            lib.trace_lm_kv_reset(c_kv)
            assert lib.trace_lm_forward(c_model, c_kv, c_tokens, seq_len, logits_scalar, 1, c_ws) == seq_len
            arr_sc = np.ctypeslib.as_array(logits_scalar).copy()

            lib.trace_cpu_set_isa(b"avx2")
            lib.trace_lm_kv_reset(c_kv)
            assert lib.trace_lm_forward(c_model, c_kv, c_tokens, seq_len, logits_avx2, 1, c_ws) == seq_len
            arr_ax = np.ctypeslib.as_array(logits_avx2).copy()

            diff_arr = np.abs(arr_sc - arr_ax)
            max_diff = float(np.max(diff_arr))
            mean_diff = float(np.mean(diff_arr))
            assert max_diff < 2.5e-2, f"Divergencia scalar vs avx2 en prefill S=64: max={max_diff:.6e}"
            assert mean_diff < 3e-3, f"Deriva media scalar vs avx2 en prefill S=64: mean={mean_diff:.6e}"
            assert int(np.argmax(arr_sc)) == int(np.argmax(arr_ax))
        finally:
            lib.trace_cpu_set_isa(b"auto")
    finally:
        if c_kv is not None:
            lib.trace_lm_kv_free(c_kv)
        if c_ws is not None:
            lib.trace_lm_workspace_free(c_ws)
        lib.trace_lm_free(c_model)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
