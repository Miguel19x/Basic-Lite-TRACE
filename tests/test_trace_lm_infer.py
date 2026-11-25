# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Suite del motor nativo C standalone para TRACELanguageModel (libtrace_lm.so y ./trace_lm)."""

import ctypes
import os
import subprocess

import numpy as np
import pytest
import torch

from tests._helpers import REPO_ROOT, load_trace_lm_lib, make_small_lm_config
from trace.lm.config import TRACELMConfig
from trace.lm.model import TRACELanguageModel
from trace.lm.tokenizer import TRACETokenizer
from trace.serialization.paged_bitcos import PagedBitCOSSerializer

CLI_BIN = str(REPO_ROOT / "trace_lm")


def _export_small_lm(tmp_path, seed: int = 42, vocab_size: int = 4096, max_seq_len: int = 256):
    torch.manual_seed(seed)
    cfg = make_small_lm_config(vocab_size=vocab_size, max_seq_len=max_seq_len)
    model = TRACELanguageModel(cfg).eval()
    bin_path = str(tmp_path / f"lm_s{seed}.bin")
    PagedBitCOSSerializer.export_paged_model(model=model, output_path=bin_path, format_type="fast")
    return cfg, model, bin_path


def test_c_lm_loader_performs_two_pread_calls(tmp_path):
    lib = load_trace_lm_lib()
    _, _, bin_path = _export_small_lm(tmp_path)

    c_model = lib.trace_lm_load(bin_path.encode())
    assert c_model is not None
    try:
        assert lib.trace_lm_get_pread_calls(c_model) == 2
        # tmp_path puede residir en tmpfs (sin soporte O_DIRECT en CI): aceptar fallback limpio (0 o 1)
        assert lib.trace_lm_get_used_odirect(c_model) in (0, 1)
    finally:
        lib.trace_lm_free(c_model)


@pytest.mark.parametrize("use_int8_kv", [0, 1])
def test_c_lm_prefill_matches_pytorch_oracle(tmp_path, use_int8_kv: int):
    lib = load_trace_lm_lib()
    cfg, model, bin_path = _export_small_lm(tmp_path)
    prompt = [15, 42, 88, 120, 200, 50, 12, 99]
    input_ids = torch.tensor([prompt], dtype=torch.long)

    c_model = lib.trace_lm_load(bin_path.encode())
    assert c_model is not None
    c_ws = c_kv = None
    try:
        c_cfg = lib.trace_lm_get_config(c_model)
        c_ws = lib.trace_lm_workspace_create(c_cfg, 256)
        c_kv = lib.trace_lm_kv_create(c_cfg, 256, use_int8_kv)
        assert c_ws is not None and c_kv is not None

        c_tokens = (ctypes.c_int * len(prompt))(*prompt)
        c_logits = (ctypes.c_float * (len(prompt) * cfg.vocab_size))()
        assert lib.trace_lm_forward(c_model, c_kv, c_tokens, len(prompt), c_logits, 0, c_ws) == len(prompt)

        with torch.no_grad():
            py_logits, _, _ = model(input_ids, use_weight_absorption=True, quantize_kv_int8=(use_int8_kv == 1))

        c_arr = np.ctypeslib.as_array(c_logits).reshape(len(prompt), cfg.vocab_size)
        py_arr = py_logits.squeeze(0).numpy()
        assert float(np.max(np.abs(c_arr - py_arr))) < 1e-4

        offset_last = (len(prompt) - 1) * cfg.vocab_size * 4
        ptr_last = ctypes.cast(ctypes.addressof(c_logits) + offset_last, ctypes.POINTER(ctypes.c_float))
        assert lib.trace_lm_sample_greedy(ptr_last, cfg.vocab_size) == int(torch.argmax(py_logits[0, -1, :]).item())
    finally:
        if c_kv is not None:
            lib.trace_lm_kv_free(c_kv)
        if c_ws is not None:
            lib.trace_lm_workspace_free(c_ws)
        lib.trace_lm_free(c_model)


def test_c_lm_incremental_decode_matches_pytorch_generate(tmp_path):
    lib = load_trace_lm_lib()
    cfg, model, bin_path = _export_small_lm(tmp_path)
    prompt = [15, 42, 88, 120, 200, 50, 12, 99]
    input_ids = torch.tensor([prompt], dtype=torch.long)

    c_model = lib.trace_lm_load(bin_path.encode())
    assert c_model is not None
    c_ws = c_kv = None
    try:
        c_cfg = lib.trace_lm_get_config(c_model)
        c_ws = lib.trace_lm_workspace_create(c_cfg, 256)
        c_kv = lib.trace_lm_kv_create(c_cfg, 256, 1)
        assert c_ws is not None and c_kv is not None

        c_tokens = (ctypes.c_int * len(prompt))(*prompt)
        c_logits = (ctypes.c_float * cfg.vocab_size)()
        assert lib.trace_lm_forward(c_model, c_kv, c_tokens, len(prompt), c_logits, 1, c_ws) == len(prompt)
        tok1 = lib.trace_lm_sample_greedy(c_logits, cfg.vocab_size)

        step2 = list(prompt) + [tok1]
        c_step2 = (ctypes.c_int * len(step2))(*step2)
        assert lib.trace_lm_forward(c_model, c_kv, c_step2, len(step2), c_logits, 1, c_ws) == 1
        assert lib.trace_lm_kv_get_last_common_prefix(c_kv) == len(prompt)
        tok2 = lib.trace_lm_sample_greedy(c_logits, cfg.vocab_size)

        with torch.no_grad():
            py_gen = model.generate(input_ids, max_new_tokens=2, temperature=0.0, quantize_kv_int8=True)
        assert [tok1, tok2] == py_gen[0, -2:].tolist()
    finally:
        if c_kv is not None:
            lib.trace_lm_kv_free(c_kv)
        if c_ws is not None:
            lib.trace_lm_workspace_free(c_ws)
        lib.trace_lm_free(c_model)


def test_prefix_caching_reuses_common_prefix_without_logit_drift(tmp_path):
    lib = load_trace_lm_lib()
    cfg, _, bin_path = _export_small_lm(tmp_path, seed=99)

    prefix = torch.randint(10, 4000, (128,)).tolist()
    prompt_a = prefix + torch.randint(10, 4000, (16,)).tolist()
    prompt_b = prefix + torch.randint(10, 4000, (16,)).tolist()

    c_model = lib.trace_lm_load(bin_path.encode())
    assert c_model is not None
    c_ws = c_kv = None
    try:
        c_cfg = lib.trace_lm_get_config(c_model)
        c_ws = lib.trace_lm_workspace_create(c_cfg, 256)
        c_kv = lib.trace_lm_kv_create(c_cfg, 256, 1)
        assert c_ws is not None and c_kv is not None

        logits_a = (ctypes.c_float * cfg.vocab_size)()
        c_pa = (ctypes.c_int * len(prompt_a))(*prompt_a)
        assert lib.trace_lm_forward(c_model, c_kv, c_pa, len(prompt_a), logits_a, 1, c_ws) == len(prompt_a)
        assert lib.trace_lm_kv_get_last_common_prefix(c_kv) == 0

        logits_b_cached = (ctypes.c_float * cfg.vocab_size)()
        c_pb = (ctypes.c_int * len(prompt_b))(*prompt_b)
        assert lib.trace_lm_forward(c_model, c_kv, c_pb, len(prompt_b), logits_b_cached, 1, c_ws) == 16
        assert lib.trace_lm_kv_get_last_common_prefix(c_kv) == 128

        lib.trace_lm_kv_reset(c_kv)
        logits_b_uncached = (ctypes.c_float * cfg.vocab_size)()
        assert lib.trace_lm_forward(c_model, c_kv, c_pb, len(prompt_b), logits_b_uncached, 1, c_ws) == len(prompt_b)

        diff = float(np.max(np.abs(np.ctypeslib.as_array(logits_b_cached) - np.ctypeslib.as_array(logits_b_uncached))))
        assert diff < 1e-5
    finally:
        if c_kv is not None:
            lib.trace_lm_kv_free(c_kv)
        if c_ws is not None:
            lib.trace_lm_workspace_free(c_ws)
        lib.trace_lm_free(c_model)


def test_prefix_caching_handles_identical_repeat_prompt(tmp_path):
    lib = load_trace_lm_lib()
    cfg, _, bin_path = _export_small_lm(tmp_path, seed=101)
    prompt = torch.randint(10, 4000, (32,)).tolist()

    c_model = lib.trace_lm_load(bin_path.encode())
    assert c_model is not None
    c_ws = c_kv = None
    try:
        c_cfg = lib.trace_lm_get_config(c_model)
        c_ws = lib.trace_lm_workspace_create(c_cfg, 256)
        c_kv = lib.trace_lm_kv_create(c_cfg, 256, 1)
        assert c_ws is not None and c_kv is not None

        c_p = (ctypes.c_int * len(prompt))(*prompt)
        logits1 = (ctypes.c_float * cfg.vocab_size)()
        logits2 = (ctypes.c_float * cfg.vocab_size)()
        assert lib.trace_lm_forward(c_model, c_kv, c_p, len(prompt), logits1, 1, c_ws) == len(prompt)
        assert lib.trace_lm_forward(c_model, c_kv, c_p, len(prompt), logits2, 1, c_ws) == 1
        assert lib.trace_lm_kv_get_last_common_prefix(c_kv) == len(prompt) - 1
        assert float(np.max(np.abs(np.ctypeslib.as_array(logits1) - np.ctypeslib.as_array(logits2)))) < 1e-5
    finally:
        if c_kv is not None:
            lib.trace_lm_kv_free(c_kv)
        if c_ws is not None:
            lib.trace_lm_workspace_free(c_ws)
        lib.trace_lm_free(c_model)


def test_c_lm_loader_rejects_invalid_binaries(tmp_path):
    lib = load_trace_lm_lib()
    corrupt_bin = tmp_path / "corrupt.bin"
    corrupt_bin.write_bytes(b"NOT_A_VALID_TRACE_BIN_HEADER_1234567890")
    assert lib.trace_lm_load(str(corrupt_bin).encode()) is None
    assert lib.trace_lm_load(str(tmp_path / "missing.bin").encode()) is None


def test_c_lm_forward_validates_token_length_bounds(tmp_path):
    lib = load_trace_lm_lib()
    cfg, _, bin_path = _export_small_lm(tmp_path, seed=77, vocab_size=512, max_seq_len=64)

    c_model = lib.trace_lm_load(bin_path.encode())
    assert c_model is not None
    c_ws = c_kv = None
    try:
        c_cfg = lib.trace_lm_get_config(c_model)
        c_ws = lib.trace_lm_workspace_create(c_cfg, 64)
        c_kv = lib.trace_lm_kv_create(c_cfg, 64, 1)
        logits = (ctypes.c_float * cfg.vocab_size)()

        dummy = (ctypes.c_int * 1)(1)
        assert lib.trace_lm_forward(c_model, c_kv, dummy, 0, logits, 1, c_ws) == 0

        over_tokens = (ctypes.c_int * (cfg.max_seq_len + 1))(*([1] * (cfg.max_seq_len + 1)))
        assert lib.trace_lm_forward(c_model, c_kv, over_tokens, cfg.max_seq_len + 1, logits, 1, c_ws) < 0
    finally:
        if c_kv is not None:
            lib.trace_lm_kv_free(c_kv)
        if c_ws is not None:
            lib.trace_lm_workspace_free(c_ws)
        lib.trace_lm_free(c_model)


def test_c_lm_scalar_matches_avx2(tmp_path):
    lib = load_trace_lm_lib()
    cfg, _, bin_path = _export_small_lm(tmp_path, seed=77, vocab_size=512, max_seq_len=64)
    prompt = [5, 17, 42, 99, 128, 255]

    c_model = lib.trace_lm_load(bin_path.encode())
    assert c_model is not None
    c_ws = c_kv = None
    try:
        c_cfg = lib.trace_lm_get_config(c_model)
        c_ws = lib.trace_lm_workspace_create(c_cfg, 64)
        c_kv = lib.trace_lm_kv_create(c_cfg, 64, 1)
        c_tokens = (ctypes.c_int * len(prompt))(*prompt)
        logits_scalar = (ctypes.c_float * cfg.vocab_size)()
        logits_avx2 = (ctypes.c_float * cfg.vocab_size)()

        try:
            lib.trace_cpu_set_isa(b"scalar")
            assert lib.trace_cpu_active_isa().decode("utf-8") == "scalar"
            lib.trace_lm_kv_reset(c_kv)
            assert lib.trace_lm_forward(c_model, c_kv, c_tokens, len(prompt), logits_scalar, 1, c_ws) == len(prompt)
            arr_scalar = np.ctypeslib.as_array(logits_scalar).copy()

            lib.trace_cpu_set_isa(b"avx2")
            lib.trace_lm_kv_reset(c_kv)
            assert lib.trace_lm_forward(c_model, c_kv, c_tokens, len(prompt), logits_avx2, 1, c_ws) == len(prompt)
            arr_avx2 = np.ctypeslib.as_array(logits_avx2).copy()

            assert float(np.max(np.abs(arr_scalar - arr_avx2))) < 1e-4
        finally:
            lib.trace_cpu_set_isa(b"auto")
    finally:
        if c_kv is not None:
            lib.trace_lm_kv_free(c_kv)
        if c_ws is not None:
            lib.trace_lm_workspace_free(c_ws)
        lib.trace_lm_free(c_model)


def _export_cli_artifacts(tmp_path):
    torch.manual_seed(123)
    tok = TRACETokenizer.train(
        ["Había una vez en un bosque digital donde los modelos ligeros razonan."],
        target_vocab_size=320,
        min_freq=1,
    )
    tok_path = str(tmp_path / "cli_tok.bin")
    tok.save(tok_path)

    cfg = TRACELMConfig(
        vocab_size=max(512, tok.vocab_size),
        max_seq_len=64,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        kv_lora_rank=32,
        qk_nope_head_dim=24,
        qk_rope_head_dim=16,
        v_head_dim=24,
        tie_word_embeddings=True,
    )
    model = TRACELanguageModel(cfg).eval()
    model_path = str(tmp_path / "cli_lm.bin")
    PagedBitCOSSerializer.export_paged_model(model=model, output_path=model_path, format_type="fast")
    return model_path, tok_path


def test_cli_greedy_generation_is_deterministic(tmp_path):
    assert os.path.exists(CLI_BIN), f"No se encontró binario {CLI_BIN}"
    model_path, tok_path = _export_cli_artifacts(tmp_path)
    cmd = [
        CLI_BIN, "--model", model_path, "--tokenizer", tok_path,
        "--prompt", "Había una vez", "--max_tokens", "4", "--temp", "0.0",
    ]
    run1 = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    run2 = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

    assert run1.returncode == 0, run1.stderr
    assert run2.returncode == 0, run2.stderr
    assert "Había una vez" in run1.stdout
    # Filtrar líneas de telemetría con tiempos/semilla variables y comparar el texto generado
    skip_tokens = ("ms", "tok/s", "Seed:", "Throughput:")
    gen_lines_1 = [ln for ln in run1.stdout.splitlines() if not any(k in ln for k in skip_tokens)]
    gen_lines_2 = [ln for ln in run2.stdout.splitlines() if not any(k in ln for k in skip_tokens)]
    assert gen_lines_1 == gen_lines_2


@pytest.mark.parametrize("kind", ["missing", "corrupt"])
def test_cli_returns_nonzero_on_invalid_model(tmp_path, kind: str):
    assert os.path.exists(CLI_BIN), f"No se encontró binario {CLI_BIN}"
    _, tok_path = _export_cli_artifacts(tmp_path)
    bad_model = tmp_path / f"{kind}_model.bin"
    if kind == "corrupt":
        bad_model.write_bytes(b"BAD_MAGIC_HEADER")

    res = subprocess.run(
        [CLI_BIN, "--model", str(bad_model), "--tokenizer", tok_path, "--prompt", "Hola", "--max_tokens", "2"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert res.returncode != 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
