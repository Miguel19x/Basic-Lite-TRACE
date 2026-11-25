# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Suite de paridad bit a bit Python <-> C, round-trip y casos borde de TRACETokenizer."""

import ctypes
import random

import pytest

from tests._helpers import load_trace_tokenizer_lib
from trace.lm.tokenizer import (
    SPECIAL_BOS,
    SPECIAL_EOS,
    SPECIAL_MASK,
    SPECIAL_PAD,
    TRACETokenizer,
    pre_tokenize,
)

TEST_SAMPLES = [
    "3.1415926535",
    "x = 42 + 108 * 3",
    "La aceleración de inferencia alcanzó 4.02x en Intel i3-1115G4.",
    "El algoritmo de atención MLA reduce la memoria KV entre 4x y 8x.",
    "def forward(self, x: torch.Tensor) -> torch.Tensor:\n    return self.sub_ln(x)\n",
    '#include <stdio.h>\nint main() { printf("Hola Mundo\\n"); return 0; }',
    "¿Cómo estás? ¡Muy bien! Año 2026: comunicación cuántica y redes neuronales.",
    "Modelos de lenguaje ligeros 🚀 con cuantización W1.58A8 🔥 y Causal Core.",
    "A" * 350,
]

CORPUS_MIXED = [
    "El modelo de lenguaje Basic-Lite TRACE opera sobre CPU.",
    "La arquitectura de atención MLA y la cuantización W1.58A8.",
    "def calcular_gradiente(x, y):\n    return x * y + 42\n",
    "comunicación constante y verificación formal en el año 2026.",
    "Aceleración 4.02x lograda con O_DIRECT y kernel AVX2.",
    "A" * 100 + " " + "A" * 100,
]

CORPUS_BILINGUAL = [
    "El gato está durmiendo plácidamente sobre el teclado en la oficina.",
    "The quick brown fox jumps over the lazy sleeping dog in the forest.",
    "La optimización de hiperparámetros aceleró la convergencia del modelo.",
    "BitNet W1.58 quantization reduces linear projections to ternary addition and subtraction.",
    "La memoria paginada de 4096 bytes garantiza lecturas directas del subsistema NVMe.",
    "Natural language processing bridges human concepts with algorithmic representations.",
    "Validación exhaustiva de paridad bilingüe en inglés y español para el Hito 3.6b.",
]


def _generate_property_strings(seed: int = 2026, count: int = 25) -> list[str]:
    """Generador determinista de cadenas UTF-8 con dígitos, espacios, código y emojis."""
    rng = random.Random(seed)
    alphabet = list("abcXYZ0123456789 \n\t:._-+/()[]{}áéíóúñ¿¡🚀🔥")
    return ["".join(rng.choice(alphabet) for _ in range(rng.randint(1, 80))) for _ in range(count)]


def _open_c_tokenizer(tmp_path, corpus: list[str], target_vocab_size: int):
    c_lib = load_trace_tokenizer_lib()
    tok = TRACETokenizer.train(corpus, target_vocab_size=target_vocab_size, min_freq=1)
    bin_path = str(tmp_path / f"tok_{target_vocab_size}.bin")
    tok.save(bin_path)
    tok_loaded = TRACETokenizer(bin_path)

    c_tok_buf = ctypes.create_string_buffer(c_lib.trace_tokenizer_sizeof())
    assert c_lib.trace_tokenizer_load(c_tok_buf, bin_path.encode("utf-8")) == 0
    return c_lib, tok, tok_loaded, c_tok_buf


def test_digits_encode_as_isolated_ascii_bytes():
    tok = TRACETokenizer()
    assert tok.encode("1234567890") == [ord(c) for c in "1234567890"]

    for sample in ("12345", "item42", "7 + 8 = 15", " 42 ", "temp: 36.5C"):
        raw = sample.encode("utf-8")
        segments = list(pre_tokenize(raw))
        assert b"".join(segments) == raw
        for seg in segments:
            if any(48 <= b <= 57 for b in seg):
                assert len(seg) == 1, f"Dígito no aislado en segmento: {seg!r}"


@pytest.mark.parametrize(
    "raw,expected",
    [
        (b" hola mundo", [b" hola", b" mundo"]),
        (b" 42", [b" ", b"4", b"2"]),
        (b"    x", [b"    ", b"x"]),
    ],
)
def test_pre_tokenize_preserves_space_prefix_rules(raw: bytes, expected: list[bytes]):
    assert list(pre_tokenize(raw)) == expected


def test_empty_input_encodes_to_empty(tmp_path):
    c_lib, _, tok, c_tok_buf = _open_c_tokenizer(tmp_path, CORPUS_MIXED, 320)
    try:
        out_c_buf = (ctypes.c_int32 * 64)()
        out_text_buf = ctypes.create_string_buffer(64)

        assert tok.encode("") == []
        assert c_lib.trace_tokenizer_encode(c_tok_buf, b"", 0, out_c_buf, 64) == 0
        dec_len = c_lib.trace_tokenizer_decode(c_tok_buf, out_c_buf, 0, out_text_buf, 64)
        assert dec_len == 0 and out_text_buf.raw[0] == 0
    finally:
        c_lib.trace_tokenizer_free(c_tok_buf)


def test_special_tokens_decode_to_empty(tmp_path):
    c_lib, _, tok, c_tok_buf = _open_c_tokenizer(tmp_path, CORPUS_MIXED, 320)
    try:
        special_seq = [SPECIAL_BOS, SPECIAL_PAD, SPECIAL_MASK, SPECIAL_EOS]
        assert tok.decode(special_seq) == ""

        c_spec_buf = (ctypes.c_int32 * len(special_seq))(*special_seq)
        out_text_buf = ctypes.create_string_buffer(64)
        dec_len = c_lib.trace_tokenizer_decode(c_tok_buf, c_spec_buf, len(special_seq), out_text_buf, 64)
        assert dec_len == 0 and out_text_buf.raw[0] == 0
    finally:
        c_lib.trace_tokenizer_free(c_tok_buf)


@pytest.mark.parametrize(
    "corpus,target_vocab,extra_samples",
    [
        (CORPUS_MIXED, 320, TEST_SAMPLES + _generate_property_strings(2026, 20)),
        (CORPUS_BILINGUAL, 512, CORPUS_BILINGUAL),
    ],
)
def test_bpe_roundtrip_python_matches_c_oracle(
    tmp_path, corpus: list[str], target_vocab: int, extra_samples: list[str]
):
    c_lib, tok_orig, tok_loaded, c_tok_buf = _open_c_tokenizer(tmp_path, corpus, target_vocab)
    assert tok_orig.num_merges > 0
    assert tok_loaded.num_merges == tok_orig.num_merges
    assert tok_loaded.vocab_size == tok_orig.vocab_size

    try:
        out_c_buf = (ctypes.c_int32 * 2048)()
        out_text_buf = ctypes.create_string_buffer(8192)

        for text in extra_samples:
            raw_b = text.encode("utf-8")
            py_tokens = tok_loaded.encode(text)
            c_count = c_lib.trace_tokenizer_encode(c_tok_buf, raw_b, len(raw_b), out_c_buf, 2048)
            c_tokens = [out_c_buf[i] for i in range(c_count)]

            assert py_tokens == c_tokens
            assert tok_loaded.decode(py_tokens) == text

            c_len = c_lib.trace_tokenizer_decode(c_tok_buf, out_c_buf, c_count, out_text_buf, 8192)
            assert c_len == len(raw_b)
            assert out_text_buf.raw[:c_len].decode("utf-8") == text

            wrapped = [SPECIAL_BOS] + py_tokens + [SPECIAL_EOS]
            assert tok_loaded.decode(wrapped) == text
            c_wrap = (ctypes.c_int32 * len(wrapped))(*wrapped)
            c_wrap_len = c_lib.trace_tokenizer_decode(c_tok_buf, c_wrap, len(wrapped), out_text_buf, 8192)
            assert c_wrap_len == len(raw_b)
            assert out_text_buf.raw[:c_wrap_len].decode("utf-8") == text
    finally:
        c_lib.trace_tokenizer_free(c_tok_buf)


def test_c_loader_rejects_missing_or_corrupt_file(tmp_path):
    c_lib = load_trace_tokenizer_lib()
    model_sz = c_lib.trace_tokenizer_sizeof()

    missing_path = str(tmp_path / "missing.trtk")
    assert c_lib.trace_tokenizer_load(ctypes.create_string_buffer(model_sz), missing_path.encode("utf-8")) < 0

    corrupt_path = tmp_path / "corrupt.trtk"
    corrupt_path.write_bytes(b"BAD_TRTK_MAGIC_12345")
    assert c_lib.trace_tokenizer_load(ctypes.create_string_buffer(model_sz), str(corrupt_path).encode("utf-8")) < 0


def test_c_encode_truncates_safely_to_buffer_limit(tmp_path):
    c_lib, _, tok, c_buf = _open_c_tokenizer(tmp_path, ["Hola mundo desde Basic-Lite TRACE 2026"], 300)
    try:
        sample = b"Hola mundo desde Basic-Lite TRACE 2026"
        full_py = tok.encode(sample.decode("utf-8"))
        assert len(full_py) > 4

        small_buf = (ctypes.c_int32 * 4)()
        trunc_cnt = c_lib.trace_tokenizer_encode(c_buf, sample, len(sample), small_buf, 4)
        assert trunc_cnt == 4
        assert [small_buf[i] for i in range(4)] == full_py[:4]
    finally:
        c_lib.trace_tokenizer_free(c_buf)


def test_c_decode_truncates_with_null_terminator(tmp_path):
    c_lib, _, tok, c_buf = _open_c_tokenizer(tmp_path, ["Hola mundo desde Basic-Lite TRACE 2026"], 300)
    try:
        full_py = tok.encode("Hola mundo desde Basic-Lite TRACE 2026")
        full_c_arr = (ctypes.c_int32 * len(full_py))(*full_py)
        small_txt = ctypes.create_string_buffer(6)

        trunc_len = c_lib.trace_tokenizer_decode(c_buf, full_c_arr, len(full_py), small_txt, 6)
        assert 0 <= trunc_len < 6
        assert small_txt.raw[trunc_len] == 0
    finally:
        c_lib.trace_tokenizer_free(c_buf)


def test_c_decode_skips_out_of_range_token_ids(tmp_path):
    c_lib, _, tok, c_buf = _open_c_tokenizer(tmp_path, ["Hola mundo desde Basic-Lite TRACE 2026"], 300)
    try:
        oob_tokens = [-5, -1, ord("O"), ord("K"), tok.vocab_size, tok.vocab_size + 999]
        oob_arr = (ctypes.c_int32 * len(oob_tokens))(*oob_tokens)
        out_oob = ctypes.create_string_buffer(64)

        oob_len = c_lib.trace_tokenizer_decode(c_buf, oob_arr, len(oob_tokens), out_oob, 64)
        assert oob_len == 2
        assert out_oob.raw[:oob_len].decode("utf-8") == "OK"
    finally:
        c_lib.trace_tokenizer_free(c_buf)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
