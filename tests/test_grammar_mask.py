# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Tests del motor C de Constrained Decoding JSON (`csrc/lm/trace_grammar.c`) bajo TRTK v1 y v2."""

from __future__ import annotations

import ctypes
import json
import struct
import tempfile
from pathlib import Path

import numpy as np

from tests._helpers import load_trace_tokenizer_lib


def _build_minimal_trtk_v1(path: Path) -> int:
    """Construye un tokenizador TRTK v1 mínimo (256 bytes + 4 especiales, 0 merges)."""
    pieces: list[bytes] = [bytes([b]) for b in range(256)]
    pieces.extend([b"<PAD>", b"<MASK>", b"<BOS>", b"<EOS>"])
    vocab_size = len(pieces)
    with open(path, "wb") as f:
        f.write(b"TRTK")
        f.write(struct.pack("<2I", 1, vocab_size))
        f.write(struct.pack("<5I", 0, 256, 257, 258, 259))
        for p in pieces:
            f.write(struct.pack("<H", len(p)))
            f.write(p)
    return vocab_size


def _build_minimal_trtk_v2(path: Path) -> int:
    """Construye un tokenizador TRTK v2 compacto (256 bytes + especiales + tokens JSON)."""
    pieces: list[bytes] = [bytes([b]) for b in range(256)]
    # 256..259 especiales básicos; 260..262 chat specials LLaMA-3
    pieces.extend([
        b"<pad>",
        b"<mask>",
        b"<bos>",
        b"<eos>",
        b"<|start_header_id|>",
        b"<|end_header_id|>",
        b"<|eot_id|>",
    ])
    # Tokens multi-carácter realistas de llamadas TIR JSON
    pieces.extend([
        b'{"tool":',
        b' "calc",',
        b' "args":',
        b' {"expr":',
        b' "2+2"}}',
        b"true",
        b"false",
        b"null",
        b"123",
    ])
    vocab_size = len(pieces)
    byte_to_id = list(range(256))

    with open(path, "wb") as f:
        f.write(b"TRTK")
        f.write(struct.pack("<2I", 2, vocab_size))
        # num_merges=0, pad=256, mask=257, bos=258, eos=259
        f.write(struct.pack("<5I", 0, 256, 257, 258, 259))
        f.write(struct.pack("<256I", *byte_to_id))
        # start_header=260, end_header=261, eot=262, reserved=0
        f.write(struct.pack("<4I", 260, 261, 262, 0))
        for p in pieces:
            f.write(struct.pack("<H", len(p)))
            f.write(p)
    return vocab_size


def test_grammar_accepts_valid_tool_call_bytes():
    lib = load_trace_tokenizer_lib()
    gram_buf = ctypes.create_string_buffer(int(lib.trace_json_grammar_sizeof()))
    lib.trace_json_grammar_init(gram_buf)

    payload = b'{"tool": "calc", "args": {"expr": "2+2", "flag": true, "n": -42.5, "arr": [1, null]}}'
    assert int(lib.trace_json_grammar_advance_bytes(gram_buf, payload, len(payload))) == 1
    assert int(lib.trace_json_grammar_is_done(gram_buf)) == 1


def test_grammar_rejects_malformed_json_bytes():
    lib = load_trace_tokenizer_lib()
    gram_buf = ctypes.create_string_buffer(int(lib.trace_json_grammar_sizeof()))
    lib.trace_json_grammar_init(gram_buf)

    bad = b'{"tool": "calc",}'
    assert int(lib.trace_json_grammar_advance_bytes(gram_buf, bad, len(bad))) == 0


def test_grammar_mask_produces_valid_json_on_trtk_v2():
    with tempfile.TemporaryDirectory(prefix="trace_gram_v2_") as tmp_dir:
        tok_path = Path(tmp_dir) / "tok_v2.bin"
        vocab_size = _build_minimal_trtk_v2(tok_path)

        lib = load_trace_tokenizer_lib()
        tok_buf = ctypes.create_string_buffer(int(lib.trace_tokenizer_sizeof()))
        assert int(lib.trace_tokenizer_load(tok_buf, str(tok_path).encode("utf-8"))) == 0
        gram_buf = ctypes.create_string_buffer(int(lib.trace_json_grammar_sizeof()))
        lib.trace_json_grammar_init(gram_buf)
        try:
            # Preferir explícitamente tokens multi-carácter JSON válidos
            logits = np.full(vocab_size, -5.0, dtype=np.float32)
            # Los 5 tokens de llamada TIR comienzan en vocab_size - 9
            seq_start = vocab_size - 9
            gen_ids: list[int] = []
            for step in range(5):
                logits[:] = -5.0
                logits[seq_start + step] = 10.0
                ptr = logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                allowed = int(lib.trace_json_grammar_mask_logits(gram_buf, tok_buf, ptr))
                assert allowed > 0
                next_id = int(np.argmax(logits))
                assert next_id == seq_start + step
                gen_ids.append(next_id)
                assert int(lib.trace_json_grammar_accept_token(gram_buf, tok_buf, next_id)) == 1

            assert int(lib.trace_json_grammar_is_done(gram_buf)) == 1
            # Tras DONE, únicamente eos (259) y eot (262) deben quedar habilitados
            logits[:] = 0.0
            ptr = logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            assert int(lib.trace_json_grammar_mask_logits(gram_buf, tok_buf, ptr)) == 2
            assert np.isfinite(logits[259]) and np.isfinite(logits[262])
            assert not np.isfinite(logits[ord("{")])

            out_buf = ctypes.create_string_buffer(512)
            arr = (ctypes.c_int32 * len(gen_ids))(*gen_ids)
            n = int(lib.trace_tokenizer_decode(tok_buf, arr, len(gen_ids), out_buf, len(out_buf)))
            parsed = json.loads(out_buf.raw[:n].decode("utf-8"))
            assert parsed == {"tool": "calc", "args": {"expr": "2+2"}}
        finally:
            lib.trace_tokenizer_free(tok_buf)


def test_grammar_mask_produces_valid_json_on_trtk_v1():
    with tempfile.TemporaryDirectory(prefix="trace_gram_v1_") as tmp_dir:
        tok_path = Path(tmp_dir) / "tok_v1.bin"
        vocab_size = _build_minimal_trtk_v1(tok_path)

        lib = load_trace_tokenizer_lib()
        tok_buf = ctypes.create_string_buffer(int(lib.trace_tokenizer_sizeof()))
        assert int(lib.trace_tokenizer_load(tok_buf, str(tok_path).encode("utf-8"))) == 0
        gram_buf = ctypes.create_string_buffer(int(lib.trace_json_grammar_sizeof()))
        lib.trace_json_grammar_init(gram_buf)
        try:
            # En TRTK v1, los IDs 0..255 corresponden directamente a cada byte individual
            target = b'{"ok":true}'
            logits = np.full(vocab_size, -5.0, dtype=np.float32)
            gen_ids: list[int] = []
            for ch in target:
                logits[:] = -5.0
                tid = int(ch)
                logits[tid] = 10.0
                ptr = logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                allowed = int(lib.trace_json_grammar_mask_logits(gram_buf, tok_buf, ptr))
                assert allowed > 0
                assert np.isfinite(logits[tid])
                gen_ids.append(tid)
                assert int(lib.trace_json_grammar_accept_token(gram_buf, tok_buf, tid)) == 1

            assert int(lib.trace_json_grammar_is_done(gram_buf)) == 1
            out_buf = ctypes.create_string_buffer(256)
            arr = (ctypes.c_int32 * len(gen_ids))(*gen_ids)
            n = int(lib.trace_tokenizer_decode(tok_buf, arr, len(gen_ids), out_buf, len(out_buf)))
            assert json.loads(out_buf.raw[:n].decode("utf-8")) == {"ok": True}
        finally:
            lib.trace_tokenizer_free(tok_buf)


if __name__ == "__main__":
    test_grammar_accepts_valid_tool_call_bytes()
    test_grammar_rejects_malformed_json_bytes()
    test_grammar_mask_produces_valid_json_on_trtk_v2()
    test_grammar_mask_produces_valid_json_on_trtk_v1()
    print("[PASS] test_grammar_mask.py (4/4)")
