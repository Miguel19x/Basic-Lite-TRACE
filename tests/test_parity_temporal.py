# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Suite de paridad temporal (prefill + 5 pasos autorregresivos) PyTorch vs libtrace_lm.so."""

import ctypes

import numpy as np
import pytest
import torch

from tests._helpers import (
    REAL_LM_BIN,
    REAL_LM_TOK,
    has_real_lm_checkpoints,
    load_trace_lm_lib,
    make_small_lm_config,
)
from trace.lm.config import TRACELMConfig
from trace.lm.model import TRACELanguageModel
from trace.lm.tokenizer import TRACETokenizer
from trace.serialization.paged_bitcos import PagedBitCOSReader, PagedBitCOSSerializer


def _assert_temporal_parity(lib, bin_path: str, py_model: TRACELanguageModel, prompt_tokens: list[int], use_int8_kv: int, tol: float):
    c_model = lib.trace_lm_load(bin_path.encode())
    assert c_model is not None
    c_ws = c_kv = None
    try:
        c_cfg = lib.trace_lm_get_config(c_model)
        c_ws = lib.trace_lm_workspace_create(c_cfg, 512)
        c_kv = lib.trace_lm_kv_create(c_cfg, 512, use_int8_kv)
        assert c_ws is not None and c_kv is not None
        vocab_size = py_model.cfg.vocab_size

        input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
        with torch.no_grad():
            py_logits, _, py_kv = py_model(
                input_ids,
                use_weight_absorption=True,
                quantize_kv_int8=(use_int8_kv == 1),
                add_sub_mode=True,
            )
        py_last = py_logits[0, -1, :].numpy()
        curr_py_tok = int(np.argmax(py_last))

        c_logits = (ctypes.c_float * vocab_size)()
        c_tokens = (ctypes.c_int * len(prompt_tokens))(*prompt_tokens)
        assert lib.trace_lm_forward(c_model, c_kv, c_tokens, len(prompt_tokens), c_logits, 1, c_ws) == len(prompt_tokens)
        c_arr0 = np.ctypeslib.as_array(c_logits)
        curr_c_tok = lib.trace_lm_sample_greedy(c_logits, vocab_size)

        assert curr_c_tok == curr_py_tok
        assert float(np.max(np.abs(c_arr0 - py_last))) < tol

        accum_tokens = list(prompt_tokens) + [curr_c_tok]
        for _ in range(5):
            with torch.no_grad():
                py_dec, _, py_kv = py_model(
                    torch.tensor([[curr_py_tok]], dtype=torch.long),
                    kv_caches=py_kv,
                    use_weight_absorption=True,
                    quantize_kv_int8=(use_int8_kv == 1),
                    add_sub_mode=True,
                )
            py_dec_arr = py_dec[0, -1, :].numpy()
            curr_py_tok = int(np.argmax(py_dec_arr))

            c_accum = (ctypes.c_int * len(accum_tokens))(*accum_tokens)
            assert lib.trace_lm_forward(c_model, c_kv, c_accum, len(accum_tokens), c_logits, 1, c_ws) == 1
            assert lib.trace_lm_kv_get_last_common_prefix(c_kv) == len(accum_tokens) - 1
            c_dec_arr = np.ctypeslib.as_array(c_logits)
            curr_c_tok = lib.trace_lm_sample_greedy(c_logits, vocab_size)

            assert curr_c_tok == curr_py_tok
            assert float(np.max(np.abs(c_dec_arr - py_dec_arr))) < tol
            accum_tokens.append(curr_c_tok)
    finally:
        if c_kv is not None:
            lib.trace_lm_kv_free(c_kv)
        if c_ws is not None:
            lib.trace_lm_workspace_free(c_ws)
        lib.trace_lm_free(c_model)


@pytest.mark.parametrize("use_int8_kv", [0, 1])
def test_synthetic_temporal_decode_parity(tmp_path, use_int8_kv: int):
    torch.manual_seed(42)
    lib = load_trace_lm_lib()
    cfg = make_small_lm_config(vocab_size=512, max_seq_len=256)
    py_model = TRACELanguageModel(cfg).eval()
    bin_path = str(tmp_path / f"synthetic_temporal_{use_int8_kv}.bin")
    PagedBitCOSSerializer.export_paged_model(model=py_model, output_path=bin_path, format_type="fast")

    prompt_tokens = [15, 42, 88, 120, 200, 50, 12, 99]
    _assert_temporal_parity(lib, bin_path, py_model, prompt_tokens, use_int8_kv=use_int8_kv, tol=2e-2)


@pytest.mark.skipif(not has_real_lm_checkpoints(), reason="Checkpoint local 45m-bi ausente.")
@pytest.mark.parametrize("use_int8_kv", [0, 1])
def test_real_checkpoint_temporal_decode_parity(use_int8_kv: int):
    lib = load_trace_lm_lib()
    reader = PagedBitCOSReader(REAL_LM_BIN)
    try:
        cfg = TRACELMConfig(**reader.get_model_config())
        py_model = TRACELanguageModel(cfg)
        reader.load_into_model(py_model)
    finally:
        reader.close()
    py_model.eval()

    tok = TRACETokenizer(REAL_LM_TOK)
    prompt_tokens = tok.encode("Once upon a time, a curious engineer")
    _assert_temporal_parity(lib, REAL_LM_BIN, py_model, prompt_tokens, use_int8_kv=use_int8_kv, tol=0.6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
