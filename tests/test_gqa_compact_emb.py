# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Paridad, determinismo y robustez para GQA (estilo BitNet-2B-4T) y embeddings compactos (FP16/INT8)."""

import ctypes
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from tests._helpers import load_trace_lm_lib, make_small_lm_config
from trace.lm.config import TRACELMConfig
from trace.lm.gqa import BitNetGQALM
from trace.lm.model import TRACELanguageModel
from trace.serialization.paged_bitcos import PagedBitCOSSerializer


def _make_small_gqa_config(
    vocab_size: int = 512,
    max_seq_len: int = 256,
    tie_word_embeddings: bool = True,
) -> TRACELMConfig:
    """Configuración GQA compacta con ratio 4:1 (8 Q / 2 KV), RoPE split-half theta=500k y FFN ReLU² gated."""
    return TRACELMConfig(
        vocab_size=vocab_size,
        pad_token_id=0,
        mask_token_id=1,
        bos_token_id=2,
        eos_token_id=3,
        max_seq_len=max_seq_len,
        d_model=256,
        n_layers=2,
        n_heads=8,
        d_ff=384,
        eps=1e-5,
        kv_lora_rank=0,
        qk_nope_head_dim=0,
        qk_rope_head_dim=32,
        v_head_dim=32,
        tie_word_embeddings=tie_word_embeddings,
        attn_type="gqa",
        n_kv_heads=2,
        head_dim=32,
        rope_theta=500000.0,
        rope_interleaved=False,
        use_gated_ffn=True,
    )


def _export_gqa_model(
    tmp_path,
    seed: int = 42,
    embedding_dtype: str = "float32",
    tie_word_embeddings: bool = True,
    max_seq_len: int = 256,
):
    torch.manual_seed(seed)
    cfg = _make_small_gqa_config(max_seq_len=max_seq_len, tie_word_embeddings=tie_word_embeddings)
    model = BitNetGQALM(cfg).eval()
    bin_path = str(tmp_path / f"gqa_s{seed}_{embedding_dtype}_t{int(tie_word_embeddings)}.bin")
    PagedBitCOSSerializer.export_paged_model(
        model=model,
        output_path=bin_path,
        format_type="fast",
        embedding_dtype=embedding_dtype,
    )
    return cfg, model, bin_path


def test_compact_embedding_file_size_shrinks_with_fp16_int8(tmp_path):
    lib = load_trace_lm_lib()
    torch.manual_seed(11)
    cfg = make_small_lm_config(vocab_size=2048, max_seq_len=128)
    model = TRACELanguageModel(cfg).eval()

    paths = {}
    sizes = {}
    expected_codes = {"float32": 0, "float16": 1, "int8": 2}
    for dt in ("float32", "float16", "int8"):
        p = str(tmp_path / f"emb_{dt}.bin")
        PagedBitCOSSerializer.export_paged_model(model=model, output_path=p, format_type="fast", embedding_dtype=dt)
        paths[dt] = p
        sizes[dt] = os.path.getsize(p)

        c_model = lib.trace_lm_load(p.encode())
        assert c_model is not None
        try:
            assert lib.trace_lm_get_tok_emb_dtype(c_model) == expected_codes[dt]
            assert lib.trace_lm_get_attn_type(c_model) == 0
        finally:
            lib.trace_lm_free(c_model)

    emb_fp32_bytes = cfg.vocab_size * cfg.d_model * 4
    # FP16 ahorra ~50% de los embeddings; INT8 ahorra ~75% (descontando scales)
    assert sizes["float32"] - sizes["float16"] >= emb_fp32_bytes // 2 - 4096
    assert sizes["float32"] - sizes["int8"] >= (emb_fp32_bytes * 3) // 4 - 16384


@pytest.mark.parametrize("embedding_dtype", ["float16", "int8"])
@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_mla_compact_embeddings_match_pytorch_oracle(tmp_path, embedding_dtype: str, tie_word_embeddings: bool):
    lib = load_trace_lm_lib()
    torch.manual_seed(23)
    cfg = make_small_lm_config(vocab_size=512, max_seq_len=128)
    cfg.tie_word_embeddings = tie_word_embeddings
    model = TRACELanguageModel(cfg).eval()
    bin_path = str(tmp_path / f"mla_{embedding_dtype}_t{int(tie_word_embeddings)}.bin")
    PagedBitCOSSerializer.export_paged_model(
        model=model, output_path=bin_path, format_type="fast", embedding_dtype=embedding_dtype
    )

    prompt = [12, 45, 78, 123, 250, 300, 19, 88]
    input_ids = torch.tensor([prompt], dtype=torch.long)

    c_model = lib.trace_lm_load(bin_path.encode())
    assert c_model is not None
    c_ws = c_kv = None
    try:
        c_cfg = lib.trace_lm_get_config(c_model)
        c_ws = lib.trace_lm_workspace_create(c_cfg, len(prompt))
        c_kv = lib.trace_lm_kv_create(c_cfg, 128, 0)
        assert c_ws is not None and c_kv is not None

        c_tokens = (ctypes.c_int * len(prompt))(*prompt)
        c_logits = (ctypes.c_float * cfg.vocab_size)()
        assert lib.trace_lm_forward(c_model, c_kv, c_tokens, len(prompt), c_logits, 1, c_ws) == len(prompt)

        with torch.no_grad():
            py_logits, _, _ = model(input_ids, use_weight_absorption=True, quantize_kv_int8=False, add_sub_mode=True)

        c_arr = np.ctypeslib.as_array(c_logits)
        py_arr = py_logits[0, -1, :].numpy()
        mean_diff = float(np.mean(np.abs(c_arr - py_arr)))
        max_diff = float(np.max(np.abs(c_arr - py_arr)))
        if embedding_dtype == "float16":
            assert mean_diff < 5e-3, f"Deriva FP16 MLA: mean={mean_diff:.6e}"
            assert max_diff < 5e-2, f"Max diff FP16 MLA: max={max_diff:.6e}"
        else:
            assert mean_diff < 5e-2, f"Deriva INT8 MLA: mean={mean_diff:.6e}"
            assert max_diff < 3e-1, f"Max diff INT8 MLA: max={max_diff:.6e}"
        assert int(np.argmax(c_arr)) == int(np.argmax(py_arr))
    finally:
        if c_kv is not None:
            lib.trace_lm_kv_free(c_kv)
        if c_ws is not None:
            lib.trace_lm_workspace_free(c_ws)
        lib.trace_lm_free(c_model)


@pytest.mark.parametrize("use_int8_kv", [0, 1])
def test_gqa_prefill_matches_pytorch_oracle(tmp_path, use_int8_kv: int):
    lib = load_trace_lm_lib()
    cfg, model, bin_path = _export_gqa_model(tmp_path, seed=42, embedding_dtype="float32")
    seq_len = 48
    prompt = [(i * 17 + 5) % cfg.vocab_size for i in range(seq_len)]
    input_ids = torch.tensor([prompt], dtype=torch.long)

    c_model = lib.trace_lm_load(bin_path.encode())
    assert c_model is not None
    assert lib.trace_lm_get_attn_type(c_model) == 1
    c_ws = c_kv = None
    try:
        c_cfg = lib.trace_lm_get_config(c_model)
        c_ws = lib.trace_lm_workspace_create(c_cfg, seq_len)
        c_kv = lib.trace_lm_kv_create(c_cfg, 256, use_int8_kv)
        assert c_ws is not None and c_kv is not None

        c_tokens = (ctypes.c_int * seq_len)(*prompt)
        c_logits = (ctypes.c_float * cfg.vocab_size)()
        assert lib.trace_lm_forward(c_model, c_kv, c_tokens, seq_len, c_logits, 1, c_ws) == seq_len

        with torch.no_grad():
            py_logits, _ = model(input_ids, quantize_kv_int8=(use_int8_kv == 1))

        c_arr = np.ctypeslib.as_array(c_logits)
        py_arr = py_logits[0, -1, :].numpy()
        diff = np.abs(c_arr - py_arr)
        if use_int8_kv:
            logit_scale = max(float(np.std(py_arr)), 1.0)
            rel_max = float(np.max(diff)) / logit_scale
            rel_mean = float(np.mean(diff)) / logit_scale
            assert rel_max < 2.5e-2, f"GQA prefill INT8 rel_max={rel_max:.6e}"
            assert rel_mean < 1e-2, f"GQA prefill INT8 rel_mean={rel_mean:.6e}"
        else:
            assert float(np.max(diff)) < 1e-3, f"GQA prefill FP32 max_diff={np.max(diff):.6e}"
            assert float(np.mean(diff)) < 2e-4, f"GQA prefill FP32 mean_diff={np.mean(diff):.6e}"
        assert int(np.argmax(c_arr)) == int(np.argmax(py_arr))
    finally:
        if c_kv is not None:
            lib.trace_lm_kv_free(c_kv)
        if c_ws is not None:
            lib.trace_lm_workspace_free(c_ws)
        lib.trace_lm_free(c_model)


@pytest.mark.parametrize("use_int8_kv", [0, 1])
def test_gqa_incremental_decode_matches_pytorch_oracle(tmp_path, use_int8_kv: int):
    lib = load_trace_lm_lib()
    cfg, model, bin_path = _export_gqa_model(tmp_path, seed=55, embedding_dtype="float32")
    prompt = [10, 42, 99, 130, 211, 7, 54, 88]

    c_model = lib.trace_lm_load(bin_path.encode())
    assert c_model is not None
    c_ws = c_kv = None
    try:
        c_cfg = lib.trace_lm_get_config(c_model)
        c_ws = lib.trace_lm_workspace_create(c_cfg, 64)
        c_kv = lib.trace_lm_kv_create(c_cfg, 128, use_int8_kv)
        assert c_ws is not None and c_kv is not None

        c_logits = (ctypes.c_float * cfg.vocab_size)()
        c_toks = (ctypes.c_int * len(prompt))(*prompt)
        assert lib.trace_lm_forward(c_model, c_kv, c_toks, len(prompt), c_logits, 1, c_ws) == len(prompt)

        with torch.no_grad():
            py_logits, py_kvs = model(
                torch.tensor([prompt], dtype=torch.long), quantize_kv_int8=(use_int8_kv == 1)
            )
        tok = lib.trace_lm_sample_greedy(c_logits, cfg.vocab_size)
        assert tok == int(torch.argmax(py_logits[0, -1, :]).item())

        seq = list(prompt)
        for _ in range(3):
            seq.append(tok)
            c_seq = (ctypes.c_int * len(seq))(*seq)
            assert lib.trace_lm_forward(c_model, c_kv, c_seq, len(seq), c_logits, 1, c_ws) == 1
            assert lib.trace_lm_kv_get_last_common_prefix(c_kv) == len(seq) - 1

            with torch.no_grad():
                py_logits, py_kvs = model(
                    torch.tensor([[tok]], dtype=torch.long),
                    past_kvs=py_kvs,
                    quantize_kv_int8=(use_int8_kv == 1),
                )
            tok = lib.trace_lm_sample_greedy(c_logits, cfg.vocab_size)
            assert tok == int(torch.argmax(py_logits[0, -1, :]).item())
    finally:
        if c_kv is not None:
            lib.trace_lm_kv_free(c_kv)
        if c_ws is not None:
            lib.trace_lm_workspace_free(c_ws)
        lib.trace_lm_free(c_model)


def test_gqa_prefill_is_bit_identical_across_thread_counts(tmp_path):
    cfg, _, bin_path = _export_gqa_model(tmp_path, seed=71, embedding_dtype="float16")
    out1 = str(tmp_path / "gqa_t1.npy")
    out2 = str(tmp_path / "gqa_t2.npy")

    snippet = (
        "import ctypes, sys, numpy as np\n"
        "from tests._helpers import load_trace_lm_lib\n"
        "bin_path, out_path, vocab = sys.argv[1], sys.argv[2], int(sys.argv[3])\n"
        "lib = load_trace_lm_lib()\n"
        "m = lib.trace_lm_load(bin_path.encode())\n"
        "c_cfg = lib.trace_lm_get_config(m)\n"
        "ws = lib.trace_lm_workspace_create(c_cfg, 64)\n"
        "kv = lib.trace_lm_kv_create(c_cfg, 128, 1)\n"
        "prompt = [(i * 13 + 7) % vocab for i in range(64)]\n"
        "c_toks = (ctypes.c_int * 64)(*prompt)\n"
        "c_out = (ctypes.c_float * vocab)()\n"
        "assert lib.trace_lm_forward(m, kv, c_toks, 64, c_out, 1, ws) == 64\n"
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
    assert np.array_equal(arr1, arr2), f"GQA no determinista entre 1 y 2 hilos: max_diff={np.max(np.abs(arr1 - arr2))}"


@pytest.mark.parametrize("embedding_dtype", ["float16", "int8"])
def test_gqa_scalar_matches_avx2_with_compact_embeddings(tmp_path, embedding_dtype: str):
    lib = load_trace_lm_lib()
    cfg, _, bin_path = _export_gqa_model(tmp_path, seed=83, embedding_dtype=embedding_dtype)
    seq_len = 32
    prompt = [(i * 11 + 2) % cfg.vocab_size for i in range(seq_len)]

    c_model = lib.trace_lm_load(bin_path.encode())
    assert c_model is not None
    c_ws = c_kv = None
    try:
        c_cfg = lib.trace_lm_get_config(c_model)
        c_ws = lib.trace_lm_workspace_create(c_cfg, seq_len)
        c_kv = lib.trace_lm_kv_create(c_cfg, 128, 1)
        c_tokens = (ctypes.c_int * seq_len)(*prompt)
        logits_sc = (ctypes.c_float * cfg.vocab_size)()
        logits_ax = (ctypes.c_float * cfg.vocab_size)()

        try:
            lib.trace_cpu_set_isa(b"scalar")
            lib.trace_lm_kv_reset(c_kv)
            assert lib.trace_lm_forward(c_model, c_kv, c_tokens, seq_len, logits_sc, 1, c_ws) == seq_len
            arr_sc = np.ctypeslib.as_array(logits_sc).copy()

            lib.trace_cpu_set_isa(b"avx2")
            lib.trace_lm_kv_reset(c_kv)
            assert lib.trace_lm_forward(c_model, c_kv, c_tokens, seq_len, logits_ax, 1, c_ws) == seq_len
            arr_ax = np.ctypeslib.as_array(logits_ax).copy()

            diff = np.abs(arr_sc - arr_ax)
            assert float(np.max(diff)) < 2.5e-2, f"Scalar vs AVX2 GQA ({embedding_dtype}): max={np.max(diff):.6e}"
            assert float(np.mean(diff)) < 5e-3, f"Scalar vs AVX2 GQA ({embedding_dtype}): mean={np.mean(diff):.6e}"
            assert int(np.argmax(arr_sc)) == int(np.argmax(arr_ax))
        finally:
            lib.trace_cpu_set_isa(b"auto")
    finally:
        if c_kv is not None:
            lib.trace_lm_kv_free(c_kv)
        if c_ws is not None:
            lib.trace_lm_workspace_free(c_ws)
        lib.trace_lm_free(c_model)


def test_gqa_rejects_invalid_kv_head_ratio(tmp_path):
    lib = load_trace_lm_lib()
    _, _, bin_path = _export_gqa_model(tmp_path, seed=91, embedding_dtype="float16")
    raw = bytearray(open(bin_path, "rb").read())
    bad_path = str(tmp_path / "bad_kv_ratio.bin")
    # Reemplazar "n_kv_heads": 2 por "n_kv_heads": 3 (8 % 3 != 0), preservando longitud exacta del JSON
    target = b'"n_kv_heads": 2'
    replacement = b'"n_kv_heads": 3'
    assert target in raw
    idx = raw.index(target)
    raw[idx : idx + len(target)] = replacement
    with open(bad_path, "wb") as f:
        f.write(raw)

    assert lib.trace_lm_load(bad_path.encode()) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
