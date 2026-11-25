# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Tests de tokenizador TRTK v2 (BPE 128K + chat LLaMA-3) e importador tools/import_bitnet.py."""

import ctypes
import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from tests._helpers import load_trace_lm_lib, load_trace_tokenizer_lib
from tools.import_bitnet import (
    _gpt2_unicode_to_bytes,
    export_hf_tokenizer_to_trtk_v2,
    import_bitnet_model,
)
from trace.lm.config import TRACELMConfig
from trace.lm.gqa import BitNetGQALM


def _bytes_to_gpt2_unicode() -> dict[int, str]:
    u2b = _gpt2_unicode_to_bytes()
    return {b: ch for ch, b in u2b.items()}


def _write_synthetic_llama3_tokenizer_json(path: Path) -> dict[str, int]:
    """Crea un tokenizer.json Byte-Level BPE con permutación GPT-2 y tokens especiales LLaMA-3."""
    b2u = _bytes_to_gpt2_unicode()
    # Permutación estilo Tiktoken/GPT-2: '!' (33) -> 0, ...
    ordered_bytes = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    for b in range(256):
        if b not in ordered_bytes:
            ordered_bytes.append(b)

    vocab: dict[str, int] = {}
    for tid, b in enumerate(ordered_bytes):
        vocab[b2u[b]] = tid

    # Añadir merges sintéticos (palabras, doble salto y dígitos de 2-3 cifras)
    next_id = 256
    merges: list[str] = []
    pairs = [
        (b2u[ord("s")], b2u[ord("y")]),
        ("sy", b2u[ord("s")]),
        (b2u[ord("u")], b2u[ord("s")]),
        ("us", b2u[ord("e")]),
        ("use", b2u[ord("r")]),
        (b2u[ord("\n")], b2u[ord("\n")]),
        (b2u[ord("2")], b2u[ord("0")]),
        ("20", b2u[ord("2")]),
        (b2u[ord(" ")], b2u[ord("h")]),
        (b2u[ord(" ") ] + "h", b2u[ord("i")]),
    ]
    for left, right in pairs:
        merged = left + right
        if merged not in vocab:
            vocab[merged] = next_id
            next_id += 1
        merges.append(f"{left} {right}")

    specials = {
        "<|begin_of_text|>": 300,
        "<|end_of_text|>": 301,
        "<|start_header_id|>": 306,
        "<|end_header_id|>": 307,
        "<|eot_id|>": 309,
    }
    added_tokens = [{"id": tid, "content": text, "special": True} for text, tid in specials.items()]

    payload = {
        "version": "1.0",
        "added_tokens": added_tokens,
        "model": {"type": "BPE", "vocab": vocab, "merges": merges},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return specials


def _write_safetensors(path: Path, tensors: dict[str, tuple[np.ndarray, str]]) -> None:
    """Escribe un archivo .safetensors válido en disco."""
    header: dict[str, dict] = {}
    data_chunks: list[bytes] = []
    offset = 0
    for name, (arr, dtype_str) in tensors.items():
        raw = np.ascontiguousarray(arr).tobytes()
        header[name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        data_chunks.append(raw)
        offset += len(raw)
    hdr_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        for chunk in data_chunks:
            f.write(chunk)


def _pack_hf_bitnet_uint8(codes_i8: np.ndarray) -> np.ndarray:
    """Empaqueta matriz ternaria int8 {-1, 0, +1} en layout hf_chunk uint8 [out//4, in]."""
    out_f, in_f = codes_i8.shape
    assert out_f % 4 == 0
    chunk = out_f // 4
    packed = np.zeros((chunk, in_f), dtype=np.uint8)
    for i in range(4):
        c = (codes_i8[i * chunk : (i + 1) * chunk, :].astype(np.int16) + 1).astype(np.uint8) & 0x03
        packed |= c << (2 * i)
    return packed


def _export_gqa_as_hf_safetensors(
    hf_dir: Path,
    model: BitNetGQALM,
    mode: str,  # "packed_u8" o "bf16"
) -> None:
    hf_dir.mkdir(parents=True, exist_ok=True)
    cfg = model.cfg
    hf_cfg = {
        "architectures": ["BitNetForCausalLM"],
        "model_type": "bitnet",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.d_model,
        "intermediate_size": cfg.d_ff,
        "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
        "num_key_value_heads": cfg.n_kv_heads,
        "head_dim": cfg.head_dim,
        "max_position_embeddings": cfg.max_seq_len,
        "rms_norm_eps": cfg.eps,
        "rope_theta": cfg.rope_theta,
        "tie_word_embeddings": cfg.tie_word_embeddings,
        "bos_token_id": cfg.bos_token_id,
        "eos_token_id": cfg.eos_token_id,
        "pad_token_id": cfg.pad_token_id,
    }
    (hf_dir / "config.json").write_text(json.dumps(hf_cfg), encoding="utf-8")

    tensors: dict[str, tuple[np.ndarray, str]] = {}

    def _to_bf16_u16(t: torch.Tensor) -> np.ndarray:
        u32 = t.detach().float().cpu().numpy().astype(np.float32).view(np.uint32)
        return (u32 >> 16).astype(np.uint16)

    if mode == "bf16":
        tensors["model.embed_tokens.weight"] = (_to_bf16_u16(model.tok_emb.weight), "BF16")
        tensors["model.norm.weight"] = (_to_bf16_u16(model.final_norm.weight), "BF16")
    else:
        tensors["model.embed_tokens.weight"] = (model.tok_emb.weight.detach().float().cpu().numpy(), "F32")
        tensors["model.norm.weight"] = (model.final_norm.weight.detach().float().cpu().numpy(), "F32")

    for i, layer in enumerate(model.layers):
        p = f"model.layers.{i}"
        tensors[f"{p}.input_layernorm.weight"] = (layer.attn_norm.weight.detach().float().cpu().numpy(), "F32")
        tensors[f"{p}.post_attention_layernorm.weight"] = (layer.ffn_norm.weight.detach().float().cpu().numpy(), "F32")
        tensors[f"{p}.self_attn.attn_sub_norm.weight"] = (layer.attn.w_out.sub_ln.weight.detach().float().cpu().numpy(), "F32")
        tensors[f"{p}.mlp.ffn_sub_norm.weight"] = (layer.ffn.down.sub_ln.weight.detach().float().cpu().numpy(), "F32")

        projs = [
            (f"{p}.self_attn.q_proj", layer.attn.w_q),
            (f"{p}.self_attn.k_proj", layer.attn.w_k),
            (f"{p}.self_attn.v_proj", layer.attn.w_v),
            (f"{p}.self_attn.o_proj", layer.attn.w_out),
            (f"{p}.mlp.gate_proj", layer.ffn.gate),
            (f"{p}.mlp.up_proj", layer.ffn.up),
            (f"{p}.mlp.down_proj", layer.ffn.down),
        ]
        for hf_name, mod in projs:
            with torch.no_grad():
                if mode == "packed_u8":
                    _, q_w, scale_w = mod.quantize_weights()
                    codes_i8 = q_w.float().cpu().numpy().astype(np.int8)
                    packed = _pack_hf_bitnet_uint8(codes_i8)
                    tensors[f"{hf_name}.weight"] = (packed, "U8")
                    tensors[f"{hf_name}.weight_scale"] = (np.array([float(scale_w.item())], dtype=np.float32), "F32")
                else:
                    tensors[f"{hf_name}.weight"] = (_to_bf16_u16(mod.weight), "BF16")

    _write_safetensors(hf_dir / "model.safetensors", tensors)


def test_trtk_v2_roundtrip_with_byte_permutation(tmp_path):
    lib = load_trace_tokenizer_lib()
    tok_json = tmp_path / "tokenizer.json"
    trtk_bin = tmp_path / "tok_v2.bin"
    _write_synthetic_llama3_tokenizer_json(tok_json)

    info = export_hf_tokenizer_to_trtk_v2(tok_json, trtk_bin)
    assert info["vocab_size"] == 310
    assert info["num_merges"] == 10

    tok_buf = ctypes.create_string_buffer(lib.trace_tokenizer_sizeof())
    assert lib.trace_tokenizer_load(tok_buf, str(trtk_bin).encode()) == 0
    try:
        sample = "user hi! 2026\n\nIt's café".encode()
        out_ids = (ctypes.c_int32 * 128)()
        n_toks = lib.trace_tokenizer_encode(tok_buf, sample, len(sample), out_ids, 128)
        assert n_toks > 0
        # "2026" debe agrupar "202" (1 token fusionado) + "6" (1 token) bajo regla \p{N}{1,3}
        out_text = ctypes.create_string_buffer(256)
        n_bytes = lib.trace_tokenizer_decode(tok_buf, out_ids, n_toks, out_text, 256)
        assert out_text.raw[:n_bytes] == sample
    finally:
        lib.trace_tokenizer_free(tok_buf)


def test_trtk_v2_chat_template_emits_llama3_control_tokens(tmp_path):
    lib = load_trace_tokenizer_lib()
    tok_json = tmp_path / "tokenizer.json"
    trtk_bin = tmp_path / "tok_v2.bin"
    specials = _write_synthetic_llama3_tokenizer_json(tok_json)
    export_hf_tokenizer_to_trtk_v2(tok_json, trtk_bin)

    tok_buf = ctypes.create_string_buffer(lib.trace_tokenizer_sizeof())
    assert lib.trace_tokenizer_load(tok_buf, str(trtk_bin).encode()) == 0
    try:
        out_ids = (ctypes.c_int32 * 128)()
        n = lib.trace_tokenizer_format_chat_llama3(tok_buf, b"sys", b"hi", 1, out_ids, 128)
        ids = [int(out_ids[i]) for i in range(n)]
        assert ids[0] == specials["<|begin_of_text|>"]
        assert ids.count(specials["<|start_header_id|>"]) == 3
        assert ids.count(specials["<|end_header_id|>"]) == 3
        assert ids.count(specials["<|eot_id|>"]) == 2
        # Decodificar debe omitir los tokens de control y conservar el contenido legible
        out_text = ctypes.create_string_buffer(256)
        n_bytes = lib.trace_tokenizer_decode(tok_buf, out_ids, n, out_text, 256)
        decoded = out_text.raw[:n_bytes].decode("utf-8")
        assert "system\n\nsys" in decoded
        assert "user\n\nhi" in decoded
        assert decoded.endswith("assistant\n\n")
    finally:
        lib.trace_tokenizer_free(tok_buf)


def test_trtk_v2_rejects_out_of_range_byte_mapping(tmp_path):
    lib = load_trace_tokenizer_lib()
    tok_json = tmp_path / "tokenizer.json"
    trtk_bin = tmp_path / "tok_v2.bin"
    _write_synthetic_llama3_tokenizer_json(tok_json)
    export_hf_tokenizer_to_trtk_v2(tok_json, trtk_bin)

    raw = bytearray(trtk_bin.read_bytes())
    # byte_to_id[0] vive justo en el offset 32 (tras cabecera de 32 bytes)
    struct.pack_into("<I", raw, 32, 999999)
    bad_bin = tmp_path / "tok_v2_bad.bin"
    bad_bin.write_bytes(raw)

    tok_buf = ctypes.create_string_buffer(lib.trace_tokenizer_sizeof())
    assert lib.trace_tokenizer_load(tok_buf, str(bad_bin).encode()) < 0


@pytest.mark.parametrize(
    ("mode", "embedding_dtype"),
    [
        ("packed_u8", "float16"),
        ("bf16", "int8"),
    ],
)
def test_import_bitnet_matches_pytorch_oracle(tmp_path, mode: str, embedding_dtype: str):
    lib = load_trace_lm_lib()
    torch.manual_seed(101)
    cfg = TRACELMConfig(
        vocab_size=256,
        pad_token_id=0,
        mask_token_id=1,
        bos_token_id=2,
        eos_token_id=3,
        max_seq_len=128,
        d_model=128,
        n_layers=2,
        n_heads=8,
        d_ff=192,
        eps=1e-5,
        kv_lora_rank=0,
        qk_nope_head_dim=0,
        qk_rope_head_dim=16,
        v_head_dim=16,
        tie_word_embeddings=True,
        attn_type="gqa",
        n_kv_heads=2,
        head_dim=16,
        rope_theta=500000.0,
        rope_interleaved=False,
        use_gated_ffn=True,
    )
    model = BitNetGQALM(cfg).eval()
    hf_dir = tmp_path / f"hf_{mode}"
    _export_gqa_as_hf_safetensors(hf_dir, model, mode=mode)

    out_bin = tmp_path / f"bitnet_{mode}_{embedding_dtype}.bin"
    info = import_bitnet_model(hf_dir, out_bin, embedding_dtype=embedding_dtype)
    assert info["n_layers"] == 2
    assert info["embedding_dtype"] == embedding_dtype

    c_model = lib.trace_lm_load(str(out_bin).encode())
    assert c_model is not None
    c_ws = c_kv = None
    try:
        c_cfg = lib.trace_lm_get_config(c_model)
        prompt = [5, 17, 42, 88, 120, 200, 31, 64]
        c_ws = lib.trace_lm_workspace_create(c_cfg, len(prompt))
        c_kv = lib.trace_lm_kv_create(c_cfg, 128, 0)
        assert c_ws is not None and c_kv is not None

        c_tokens = (ctypes.c_int * len(prompt))(*prompt)
        c_logits = (ctypes.c_float * cfg.vocab_size)()
        assert lib.trace_lm_forward(c_model, c_kv, c_tokens, len(prompt), c_logits, 1, c_ws) == len(prompt)

        with torch.no_grad():
            py_logits, _ = model(torch.tensor([prompt], dtype=torch.long), quantize_kv_int8=False)

        c_arr = np.ctypeslib.as_array(c_logits)
        py_arr = py_logits[0, -1, :].numpy()
        logit_scale = max(float(np.std(py_arr)), 1.0)
        rel_mean = float(np.mean(np.abs(c_arr - py_arr))) / logit_scale
        if embedding_dtype == "float16":
            assert rel_mean < 1.5e-2, f"Deriva import {mode}/fp16: rel_mean={rel_mean:.6e}"
        else:
            assert rel_mean < 5e-2, f"Deriva import {mode}/int8: rel_mean={rel_mean:.6e}"
        assert int(np.argmax(c_arr)) == int(np.argmax(py_arr))
    finally:
        if c_kv is not None:
            lib.trace_lm_kv_free(c_kv)
        if c_ws is not None:
            lib.trace_lm_workspace_free(c_ws)
        lib.trace_lm_free(c_model)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
