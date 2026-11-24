# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Suite de Paridad y Validación Estricta para TRACETokenizer (Hito 3.1):
1. Aislamiento estricto de dígitos (0..9 = 1 byte) y prefijo de espacio único.
2. Round-trip decode(encode(x)) == x sobre Español, Inglés, C, Python y Emojis.
3. Paridad bit a bit 100% en encode Y decode entre Python y C (ctypes).
4. Prueba de cadena vacía, tokens especiales y tok.num_merges tras load().
5. Prueba de segmento >300 bytes para certificar el fallback dinámico en C.
6. Prueba de entrenamiento BPE determinista con train().
"""

import ctypes
import os
import tempfile
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
    "#include <stdio.h>\nint main() { printf(\"Hola Mundo\\n\"); return 0; }",
    "¿Cómo estás? ¡Muy bien! Año 2026: comunicación cuántica y redes neuronales.",
    "Modelos de lenguaje ligeros 🚀 con cuantización W1.58A8 🔥 y Causal Core.",
    # Segmento continuo de >300 bytes sin espacios para certificar fallback dinámico
    "A" * 350,
]


def load_c_tokenizer_lib():
    so_path = os.path.abspath("libtrace_tokenizer.so")
    assert os.path.exists(so_path), f"El binario {so_path} no existe. Compila con make primero."
    lib = ctypes.CDLL(so_path)

    lib.trace_tokenizer_sizeof.restype = ctypes.c_size_t
    lib.trace_tokenizer_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.trace_tokenizer_load.restype = ctypes.c_int
    lib.trace_tokenizer_free.argtypes = [ctypes.c_void_p]
    lib.trace_tokenizer_free.restype = None

    lib.trace_tokenizer_encode.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
    ]
    lib.trace_tokenizer_encode.restype = ctypes.c_int

    lib.trace_tokenizer_decode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.trace_tokenizer_decode.restype = ctypes.c_int

    return lib


def test_digit_isolation_and_space_prefix():
    tok = TRACETokenizer()
    assert tok.encode("1234567890") == [ord(c) for c in "1234567890"], (
        "Los dígitos 0..9 deben mapear exactamente a sus bytes ASCII 48..57"
    )

    for sample in ("12345", "item42", "7 + 8 = 15", " 42 ", "temp: 36.5C"):
        raw = sample.encode("utf-8")
        segments = list(pre_tokenize(raw))
        assert b"".join(segments) == raw
        for seg in segments:
            if any(48 <= b <= 57 for b in seg):
                assert len(seg) == 1, f"Dígito no aislado en segmento: {seg}"

    # Verificación de prefijo de espacio único
    seg_hello = list(pre_tokenize(b" hola mundo"))
    assert seg_hello == [b" hola", b" mundo"], f"Prefijo incorrecto: {seg_hello}"

    # Verificación de espacio antes de dígito: debe quedar aislado
    seg_space_digit = list(pre_tokenize(b" 42"))
    assert seg_space_digit == [b" ", b"4", b"2"], f"Espacio antes de dígito incorrecto: {seg_space_digit}"

    # Verificación de corridas de múltiples espacios intactas (código)
    seg_indent = list(pre_tokenize(b"    x"))
    assert seg_indent == [b"    ", b"x"], f"Indentación alterada: {seg_indent}"


def test_python_and_c_parity_and_roundtrip():
    c_lib = load_c_tokenizer_lib()

    training_corpus = [
        "El modelo de lenguaje Basic-Lite TRACE opera sobre CPU.",
        "La arquitectura de atención MLA y la cuantización W1.58A8.",
        "def calcular_gradiente(x, y):\n    return x * y + 42\n",
        "comunicación constante y verificación formal en el año 2026.",
        "Aceleración 4.02x lograda con O_DIRECT y kernel AVX2.",
        "A" * 100 + " " + "A" * 100,
    ]

    tok = TRACETokenizer.train(training_corpus, target_vocab_size=320, min_freq=1)
    num_merges_trained = tok.num_merges
    assert num_merges_trained > 0, "No se registraron merges durante el entrenamiento"

    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = os.path.join(tmpdir, "test_tok_trained.bin")
        tok.save(bin_path)

        # Cargar en Python y verificar num_merges
        tok_loaded = TRACETokenizer(bin_path)
        assert tok_loaded.num_merges == num_merges_trained, (
            f"Violación: tok_loaded.num_merges ({tok_loaded.num_merges}) != {num_merges_trained}"
        )

        # Cargar en C usando trace_tokenizer_sizeof()
        model_sz = c_lib.trace_tokenizer_sizeof()
        c_tok_buf = ctypes.create_string_buffer(model_sz)
        ret = c_lib.trace_tokenizer_load(c_tok_buf, bin_path.encode("utf-8"))
        assert ret == 0, f"Error cargando tokenizer en C: {ret}"

        out_c_buf = (ctypes.c_int32 * 2048)()
        out_text_buf = ctypes.create_string_buffer(8192)

        # 1. Prueba de cadena vacía
        py_empty = tok.encode("")
        assert py_empty == [], "Encode Python de cadena vacía debe ser lista vacía"
        c_empty_cnt = c_lib.trace_tokenizer_encode(c_tok_buf, b"", 0, out_c_buf, 2048)
        assert c_empty_cnt == 0, "Encode C de cadena vacía debe retornar 0 tokens"
        c_empty_dec_len = c_lib.trace_tokenizer_decode(c_tok_buf, out_c_buf, 0, out_text_buf, 8192)
        assert c_empty_dec_len == 0 and out_text_buf.raw[0] == 0, "Decode C de lista vacía debe retornar 0 bytes y null terminator"

        # 2. Prueba de tokens especiales (deben ser omitidos en decode)
        special_seq = [SPECIAL_BOS, SPECIAL_PAD, SPECIAL_MASK, SPECIAL_EOS]
        py_spec_dec = tok.decode(special_seq)
        assert py_spec_dec == "", f"Decode de tokens especiales debe ser vacío, se obtuvo: '{py_spec_dec}'"
        c_spec_buf = (ctypes.c_int32 * len(special_seq))(*special_seq)
        c_spec_dec_len = c_lib.trace_tokenizer_decode(c_tok_buf, c_spec_buf, len(special_seq), out_text_buf, 8192)
        assert c_spec_dec_len == 0 and out_text_buf.raw[0] == 0, "Decode C de especiales debe ser vacío"

        # 3. Pruebas de paridad sobre el banco completo TEST_SAMPLES
        for text in TEST_SAMPLES:
            raw_b = text.encode("utf-8")

            # Encode en Python
            py_tokens = tok.encode(text)

            # Encode en C
            c_count = c_lib.trace_tokenizer_encode(c_tok_buf, raw_b, len(raw_b), out_c_buf, 2048)
            c_tokens = [out_c_buf[i] for i in range(c_count)]

            # Paridad estricta bit a bit en tokens
            assert py_tokens == c_tokens, (
                f"Discrepancia encode Py vs C en '{text[:30]}...':\n"
                f"Py: {py_tokens[:10]}... (len={len(py_tokens)})\n"
                f"C : {c_tokens[:10]}... (len={len(c_tokens)})"
            )

            # Decode en Python
            py_decoded = tok.decode(py_tokens)
            assert py_decoded == text, f"Fallo round-trip Py: '{text[:20]}' != '{py_decoded[:20]}'"

            # Decode en C
            c_text_len = c_lib.trace_tokenizer_decode(c_tok_buf, out_c_buf, c_count, out_text_buf, 8192)
            assert c_text_len == len(raw_b), (
                f"Fallo longitud decode C: {c_text_len} != {len(raw_b)} en muestra '{text[:20]}'"
            )
            c_decoded = out_text_buf.raw[:c_text_len].decode("utf-8")
            assert c_decoded == text, f"Fallo round-trip C: '{text[:20]}' != '{c_decoded[:20]}'"

            # Decode con BOS y EOS encapsulados
            tokens_wrapped = [SPECIAL_BOS] + py_tokens + [SPECIAL_EOS]
            assert tok.decode(tokens_wrapped) == text, "Decode Python con BOS/EOS debe ignorar especiales"
            c_wrap_buf = (ctypes.c_int32 * len(tokens_wrapped))(*tokens_wrapped)
            c_wrap_len = c_lib.trace_tokenizer_decode(c_tok_buf, c_wrap_buf, len(tokens_wrapped), out_text_buf, 8192)
            assert c_wrap_len == len(raw_b)
            assert out_text_buf.raw[:c_wrap_len].decode("utf-8") == text, "Decode C con BOS/EOS debe ignorar especiales"

        c_lib.trace_tokenizer_free(c_tok_buf)
        print("✓ Paridad bit a bit 100% Py vs C en encode/decode, round-trip, casos borde y especiales superados.")


def test_bilingual_large_vocab_roundtrip():
    c_lib = load_c_tokenizer_lib()
    bilingual_corpus = [
        "El gato está durmiendo plácidamente sobre el teclado en la oficina.",
        "The quick brown fox jumps over the lazy sleeping dog in the forest.",
        "La optimización de hiperparámetros aceleró la convergencia del modelo.",
        "BitNet W1.58 quantization reduces linear projections to ternary addition and subtraction.",
        "La memoria paginada de 4096 bytes garantiza lecturas directas del subsistema NVMe.",
        "Natural language processing bridges human concepts with algorithmic representations.",
        "Validación exhaustiva de paridad bilingüe en inglés y español para el Hito 3.6b.",
    ]
    tok = TRACETokenizer.train(bilingual_corpus, target_vocab_size=512, min_freq=1)
    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = os.path.join(tmpdir, "tok_bilingual.bin")
        tok.save(bin_path)

        tok_loaded = TRACETokenizer(bin_path)
        assert tok_loaded.vocab_size == len(tok.pieces)

        model_sz = c_lib.trace_tokenizer_sizeof()
        c_buf = ctypes.create_string_buffer(model_sz)
        ret = c_lib.trace_tokenizer_load(c_buf, bin_path.encode("utf-8"))
        assert ret == 0, f"Error cargando tokenizador bilingüe en C: {ret}"

        out_c_buf = (ctypes.c_int32 * 2048)()
        out_text_buf = ctypes.create_string_buffer(8192)

        for text in bilingual_corpus:
            raw_b = text.encode("utf-8")
            py_toks = tok_loaded.encode(text)
            c_cnt = c_lib.trace_tokenizer_encode(c_buf, raw_b, len(raw_b), out_c_buf, 2048)
            c_toks = [out_c_buf[i] for i in range(c_cnt)]
            assert py_toks == c_toks, f"Fallo paridad bilingüe: {text}"
            py_dec = tok_loaded.decode(py_toks)
            assert py_dec == text, f"Fallo decode bilingüe Py: {text}"

            c_dec_len = c_lib.trace_tokenizer_decode(c_buf, out_c_buf, c_cnt, out_text_buf, 8192)
            assert c_dec_len == len(raw_b)
            assert out_text_buf.raw[:c_dec_len].decode("utf-8") == text

        c_lib.trace_tokenizer_free(c_buf)
        print("✓ Paridad bilingüe de vocabulario ampliado superada exitosamente.")


if __name__ == "__main__":
    test_digit_isolation_and_space_prefix()
    test_python_and_c_parity_and_roundtrip()
    test_bilingual_large_vocab_roundtrip()
    print("✓ Todas las pruebas de paridad del tokenizador pasaron exitosamente.")
