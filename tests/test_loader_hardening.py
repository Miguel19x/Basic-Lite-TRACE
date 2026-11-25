# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Tests deterministas de fuzzing ligero y endurecimiento para los 3 cargadores binarios C y el lector Python."""

import ctypes
import random
import struct

import pytest
import torch

from tests._helpers import (
    load_trace_infer_lib,
    load_trace_lm_lib,
    load_trace_tokenizer_lib,
    make_small_lm_config,
)
from trace.lm.model import TRACELanguageModel
from trace.lm.tokenizer import TRACETokenizer
from trace.serialization.paged_bitcos import PagedBitCOSReader, PagedBitCOSSerializer


def _bitcos_header(
    magic: bytes = b"PBITCOS1",
    page_size: int = 4096,
    fmt_code: int = 1,
    total_pages: int = 1,
    idx_off: int = 64,
    exp_off: int = 64,
    res_off: int = 64,
    pgs_off: int = 4096,
    total_sz: int = 8192,
) -> bytes:
    return struct.pack(
        "<8sIIIQQQQQ",
        magic,
        page_size,
        fmt_code,
        total_pages,
        idx_off,
        exp_off,
        res_off,
        pgs_off,
        total_sz,
    )


def _malformed_bitcos_payloads() -> list[tuple[str, bytes]]:
    return [
        ("empty", b""),
        ("short_8b", b"PBITCOS1"),
        ("short_55b", b"PBITCOS1" + bytes(47)),
        ("zero_page_size", _bitcos_header(page_size=0) + bytes(128)),
        ("unaligned_pages_offset", _bitcos_header(pgs_off=100, total_sz=256) + bytes(192)),
        ("inverted_offsets", _bitcos_header(idx_off=200, exp_off=100, res_off=150, pgs_off=4096) + bytes(4096)),
        ("huge_total_pages", _bitcos_header(total_pages=0xFFFFFFFF, pgs_off=4096, total_sz=4096) + bytes(4096)),
        ("truncated_pages_region", _bitcos_header(total_pages=10, pgs_off=4096, total_sz=8192) + bytes(4096)),
        (
            "unclosed_json_array",
            _bitcos_header(idx_off=64, exp_off=96, res_off=96, pgs_off=4096, total_sz=8192)
            + b'{"layers.0.attn.w_q":{"page_ids":[1, 2'
            + bytes(4096),
        ),
    ]


@pytest.mark.parametrize(
    "case_name,payload",
    [
        ("empty", b""),
        ("short_4b", b"TRTK"),
        ("short_12b", struct.pack("<III", 0x4B545254, 1, 300)),
        ("bad_version_zero", struct.pack("<IIIIIIII", 0x4B545254, 0, 300, 10, 256, 257, 258, 259)),
        ("vocab_too_small", struct.pack("<IIIIIIII", 0x4B545254, 1, 259, 0, 256, 257, 258, 259)),
        ("vocab_overflow", struct.pack("<IIIIIIII", 0x4B545254, 1, 0xFFFFFFFF, 10, 256, 257, 258, 259)),
        ("merges_exceed_vocab", struct.pack("<IIIIIIII", 0x4B545254, 1, 300, 100, 256, 257, 258, 259)),
        ("truncated_merges_body", struct.pack("<IIIIIIII", 0x4B545254, 1, 300, 20, 256, 257, 258, 259) + bytes(8)),
    ],
)
def test_tokenizer_loader_rejects_malformed_headers(tmp_path, case_name: str, payload: bytes):
    lib = load_trace_tokenizer_lib()
    bad_path = tmp_path / f"tok_{case_name}.trtk"
    bad_path.write_bytes(payload)

    buf = ctypes.create_string_buffer(lib.trace_tokenizer_sizeof())
    ret = lib.trace_tokenizer_load(buf, str(bad_path).encode("utf-8"))
    if ret == 0:
        lib.trace_tokenizer_free(buf)
    assert ret < 0, f"Cargador tokenizer debía rechazar '{case_name}', retornó {ret}"


def test_tokenizer_loader_survives_seeded_header_mutations(tmp_path):
    lib = load_trace_tokenizer_lib()
    tok = TRACETokenizer.train(["Basic-Lite TRACE tokenizer fuzzing corpus 2026"], target_vocab_size=280, min_freq=1)
    valid_path = tmp_path / "valid.trtk"
    tok.save(str(valid_path))
    raw = valid_path.read_bytes()

    rng = random.Random(2026)
    for i in range(25):
        mut = bytearray(raw)
        for _ in range(rng.randint(1, 4)):
            pos = rng.randint(0, min(len(mut) - 1, 64))
            mut[pos] ^= 1 << rng.randint(0, 7)
        if i % 5 == 0:
            mut = mut[: rng.randint(0, min(len(mut), 40))]
        p = tmp_path / f"mut_tok_{i}.trtk"
        p.write_bytes(mut)
        buf = ctypes.create_string_buffer(lib.trace_tokenizer_sizeof())
        ret = lib.trace_tokenizer_load(buf, str(p).encode("utf-8"))
        if ret == 0:
            lib.trace_tokenizer_free(buf)


@pytest.mark.parametrize("case_name,payload", _malformed_bitcos_payloads())
def test_causal_loader_rejects_malformed_bitcos(tmp_path, case_name: str, payload: bytes):
    lib = load_trace_infer_lib()
    bad_path = tmp_path / f"causal_{case_name}.bin"
    bad_path.write_bytes(payload)

    buf = ctypes.create_string_buffer(lib.trace_model_sizeof())
    ret = lib.trace_model_load_odirect(str(bad_path).encode("utf-8"), buf)
    if ret == 0:
        lib.trace_model_free(buf)
    assert ret != 0, f"trace_model_load_odirect debía rechazar '{case_name}'"


@pytest.mark.parametrize("case_name,payload", _malformed_bitcos_payloads())
def test_lm_loader_rejects_malformed_bitcos(tmp_path, case_name: str, payload: bytes):
    lib = load_trace_lm_lib()
    bad_path = tmp_path / f"lm_{case_name}.bin"
    bad_path.write_bytes(payload)

    ptr = lib.trace_lm_load(str(bad_path).encode("utf-8"))
    if ptr is not None:
        lib.trace_lm_free(ptr)
    assert ptr is None, f"trace_lm_load debía devolver NULL en '{case_name}'"


def test_lm_loader_survives_seeded_header_mutations(tmp_path):
    lib = load_trace_lm_lib()
    torch.manual_seed(42)
    cfg = make_small_lm_config(vocab_size=512, max_seq_len=64)
    model = TRACELanguageModel(cfg).eval()
    valid_path = tmp_path / "valid_lm.bin"
    PagedBitCOSSerializer.export_paged_model(model=model, output_path=str(valid_path), format_type="fast")
    raw = valid_path.read_bytes()

    rng = random.Random(1337)
    for i in range(20):
        mut = bytearray(raw)
        for _ in range(rng.randint(1, 3)):
            pos = rng.randint(0, min(len(mut) - 1, 128))
            mut[pos] ^= 1 << rng.randint(0, 7)
        p = tmp_path / f"mut_lm_{i}.bin"
        p.write_bytes(mut)
        ptr = lib.trace_lm_load(str(p).encode("utf-8"))
        if ptr is not None:
            lib.trace_lm_free(ptr)


@pytest.mark.parametrize(
    "payload",
    [
        b"PBITCOS1",
        _bitcos_header(page_size=0) + bytes(64),
        _bitcos_header(idx_off=200, exp_off=100, res_off=150, pgs_off=4096) + bytes(4096),
    ],
)
def test_python_reader_rejects_invalid_bitcos_headers(tmp_path, payload: bytes):
    bad_path = tmp_path / "bad_py_reader.bin"
    bad_path.write_bytes(payload)
    with pytest.raises(ValueError):
        PagedBitCOSReader(str(bad_path))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
