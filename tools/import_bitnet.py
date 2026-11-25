#!/usr/bin/env python3
# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Importador de checkpoints HuggingFace BitNet b1.58 (ej. microsoft/bitnet-b1.58-2B-4T
y microsoft/bitnet-b1.58-2B-4T-bf16) hacia Paged BitCOS v1.1 (trace-lm-v1) y
tokenizador binario nativo TRTK v2 (LLaMA-3 Byte-Level BPE 128K).

Diseñado para streaming capa por capa (< 1.5 GB RAM pico en modelos 2B) sin
requerir `transformers` ni `safetensors` instalados.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np

from trace.serialization.paged_bitcos import DEFAULT_PAGE_SIZE, MAGIC_PBITCOS, PagedBitCOSSerializer

TRTK_MAGIC = 0x4B545254  # "TRTK"


# =====================================================================
# Lector ligero de SafeTensors (sin dependencias externas, carga por demanda)
# =====================================================================
class SafeTensorsReader:
    """Lee tensores individuales de uno o varios archivos .safetensors sin cargar todo a RAM."""

    _DTYPE_MAP: dict[str, tuple[np.dtype, int]] = {
        "F32": (np.dtype("<f4"), 4),
        "F16": (np.dtype("<f2"), 2),
        "BF16": (np.dtype("<u2"), 2),
        "I8": (np.dtype("<i1"), 1),
        "U8": (np.dtype("<u1"), 1),
        "I32": (np.dtype("<i4"), 4),
        "I64": (np.dtype("<i8"), 8),
    }

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self._entries: dict[str, tuple[Path, int, str, list[int], int, int]] = {}
        index_path = self.model_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path, encoding="utf-8") as f:
                idx = json.load(f)
            shard_files = sorted(set(idx.get("weight_map", {}).values()))
            for shard in shard_files:
                self._index_file(self.model_dir / shard)
        else:
            single = self.model_dir / "model.safetensors"
            if not single.exists():
                raise FileNotFoundError(f"No se encontró model.safetensors ni índice en {self.model_dir}")
            self._index_file(single)

    def _index_file(self, path: Path) -> None:
        with open(path, "rb") as f:
            hdr_len_bytes = f.read(8)
            if len(hdr_len_bytes) != 8:
                raise ValueError(f"Cabecera safetensors truncada en {path}")
            hdr_len = struct.unpack("<Q", hdr_len_bytes)[0]
            hdr_raw = f.read(hdr_len)
            if len(hdr_raw) != hdr_len:
                raise ValueError(f"Metadatos JSON safetensors truncados en {path}")
            meta = json.loads(hdr_raw.decode("utf-8"))
        data_base = 8 + hdr_len
        for name, info in meta.items():
            if name == "__metadata__":
                continue
            dtype_str = info["dtype"]
            shape = [int(d) for d in info["shape"]]
            start, end = int(info["data_offsets"][0]), int(info["data_offsets"][1])
            self._entries[name] = (path, data_base, dtype_str, shape, start, end)

    def has_tensor(self, name: str) -> bool:
        return name in self._entries

    def tensor_names(self) -> list[str]:
        return list(self._entries.keys())

    def get_raw(self, name: str) -> tuple[np.ndarray, str]:
        if name not in self._entries:
            raise KeyError(f"Tensor '{name}' no encontrado en safetensors")
        path, data_base, dtype_str, shape, start, end = self._entries[name]
        if dtype_str not in self._DTYPE_MAP:
            raise ValueError(f"Tipo safetensors no soportado: {dtype_str} ({name})")
        np_dt, _ = self._DTYPE_MAP[dtype_str]
        with open(path, "rb") as f:
            f.seek(data_base + start)
            buf = f.read(end - start)
        arr = np.frombuffer(buf, dtype=np_dt).reshape(shape).copy()
        return arr, dtype_str

    def get_float32(self, name: str) -> np.ndarray:
        arr, dtype_str = self.get_raw(name)
        if dtype_str == "BF16":
            return (arr.astype(np.uint32) << 16).view(np.float32)
        return arr.astype(np.float32, copy=False)


def unpack_hf_bitnet_uint8(packed: np.ndarray, layout: str = "hf_chunk") -> np.ndarray:
    """
    Desempaqueta pesos ternarios 2-bit almacenados en uint8 por HuggingFace BitNet.
    - 'hf_chunk' (estándar transformers): 4 bloques contiguos a lo largo de dim 0.
    - 'interleaved': 4 filas consecutivas empaquetadas por byte en dim 0.
    Devuelve matriz int8 en {-1, 0, +1} de forma [packed.shape[0] * 4, packed.shape[1]].
    """
    if packed.dtype != np.uint8 or packed.ndim != 2:
        raise ValueError(f"Se esperaba matriz 2D uint8, recibido {packed.dtype} {packed.shape}")
    chunks = [(((packed >> (2 * i)) & 0x03).astype(np.int8) - np.int8(1)) for i in range(4)]
    if layout == "interleaved":
        out = np.empty((packed.shape[0] * 4, packed.shape[1]), dtype=np.int8)
        for i in range(4):
            out[i::4, :] = chunks[i]
        return out
    return np.concatenate(chunks, axis=0)


def quantize_absmean_ternary(w_f32: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, float]:
    """Cuantización ternaria absmean W1.58 estándar: scale = mean(|W|), q = clip(round(W / scale), -1, 1)."""
    scale = float(max(float(np.mean(np.abs(w_f32))), eps))
    q = np.clip(np.rint(w_f32 / scale), -1, 1).astype(np.int8)
    return q, scale


def _load_ternary_projection(
    reader: SafeTensorsReader,
    weight_key: str,
    pack_layout: str = "hf_chunk",
    invert_scale: bool = False,
) -> tuple[np.ndarray, float]:
    arr, dtype_str = reader.get_raw(weight_key)
    scale_key = f"{weight_key}_scale"
    if not reader.has_tensor(scale_key) and weight_key.endswith(".weight"):
        alt_key = weight_key[: -len(".weight")] + ".weight_scale"
        if reader.has_tensor(alt_key):
            scale_key = alt_key

    if dtype_str == "U8":
        codes = unpack_hf_bitnet_uint8(arr, layout=pack_layout)
        if not reader.has_tensor(scale_key):
            raise KeyError(f"Peso empaquetado '{weight_key}' requiere tensor de escala '{scale_key}'")
        sc_arr = reader.get_float32(scale_key)
        scale_val = float(sc_arr.reshape(-1)[0])
        if invert_scale and scale_val != 0.0:
            scale_val = 1.0 / scale_val
        return codes, scale_val

    if dtype_str == "I8" and reader.has_tensor(scale_key):
        codes = np.clip(arr, -1, 1).astype(np.int8)
        sc_arr = reader.get_float32(scale_key)
        scale_val = float(sc_arr.reshape(-1)[0])
        if invert_scale and scale_val != 0.0:
            scale_val = 1.0 / scale_val
        return codes, scale_val

    w_f32 = (arr.astype(np.uint32) << 16).view(np.float32) if dtype_str == "BF16" else arr.astype(np.float32)
    return quantize_absmean_ternary(w_f32)


def _write_paged_projection(
    name: str,
    codes_np: np.ndarray,
    scale_val: float,
    page_size: int,
    pages_bytes: io.BytesIO,
    tensor_index: list[dict[str, Any]],
    current_page_id: int,
) -> int:
    out_f, in_f = codes_np.shape
    padded_in_f = ((in_f + 31) // 32) * 32
    row_bytes = padded_in_f // 4
    rows_per_page = max(1, (page_size - 64) // row_bytes)

    tensor_pages: list[int] = []
    for r_start in range(0, out_f, rows_per_page):
        r_end = min(out_f, r_start + rows_per_page)
        chunk = codes_np[r_start:r_end, :]
        page_blob = PagedBitCOSSerializer.pack_ternary_page(
            row_start=r_start,
            out_features=out_f,
            in_features=in_f,
            scale_w=scale_val,
            q_w_chunk=chunk,
            page_size=page_size,
            format_type="fast",
        )
        pages_bytes.write(page_blob)
        tensor_pages.append(current_page_id)
        current_page_id += 1

    tensor_index.append({
        "name": name,
        "shape": [out_f, in_f],
        "is_resident": False,
        "scale": float(scale_val),
        "num_pages": len(tensor_pages),
        "page_ids": tensor_pages,
    })
    return current_page_id


def _write_resident_tensor(
    name: str,
    arr_f32: np.ndarray,
    resident_bytes: io.BytesIO,
    tensor_index: list[dict[str, Any]],
    dtype: str = "float32",
) -> None:
    if dtype == "float16":
        p_bytes = arr_f32.astype(np.float16).tobytes()
        out_dtype = "float16"
    elif dtype == "int8":
        amax = np.maximum(np.max(np.abs(arr_f32), axis=-1), 1e-6).astype(np.float32)
        scales = (amax / 127.0).astype(np.float32)
        q = np.clip(np.rint(arr_f32 / scales[:, None]), -127, 127).astype(np.int8)
        p_bytes = q.tobytes() + scales.tobytes()
        out_dtype = "int8"
    else:
        p_bytes = arr_f32.astype(np.float32, copy=False).tobytes()
        out_dtype = "float32"

    p_offset = resident_bytes.tell()
    resident_bytes.write(p_bytes)
    tensor_index.append({
        "name": name,
        "shape": list(arr_f32.shape),
        "is_resident": True,
        "offset": p_offset,
        "length": len(p_bytes),
        "dtype": out_dtype,
    })


def import_bitnet_model(
    model_dir: str | Path,
    output_bin: str | Path,
    embedding_dtype: str = "float16",
    page_size: int = DEFAULT_PAGE_SIZE,
    pack_layout: str = "hf_chunk",
    invert_scale: bool = False,
) -> dict[str, Any]:
    """Importa un directorio HuggingFace BitNet b1.58 a Paged BitCOS v1.1 (trace-lm-v1)."""
    if embedding_dtype not in ("float32", "float16", "int8"):
        raise ValueError(f"embedding_dtype inválido: {embedding_dtype!r}")

    src_dir = Path(model_dir)
    cfg_path = src_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Falta config.json en {src_dir}")
    with open(cfg_path, encoding="utf-8") as f:
        hf_cfg = json.load(f)

    vocab_size = int(hf_cfg["vocab_size"])
    d_model = int(hf_cfg["hidden_size"])
    n_layers = int(hf_cfg["num_hidden_layers"])
    n_heads = int(hf_cfg["num_attention_heads"])
    n_kv_heads = int(hf_cfg.get("num_key_value_heads", n_heads))
    head_dim = int(hf_cfg.get("head_dim", d_model // n_heads))
    d_ff = int(hf_cfg["intermediate_size"])
    max_seq_len = int(hf_cfg.get("max_position_embeddings", 4096))
    eps = float(hf_cfg.get("rms_norm_eps", 1e-5))
    rope_theta = float(hf_cfg.get("rope_theta", 500000.0))
    tie_emb = bool(hf_cfg.get("tie_word_embeddings", True))
    bos_id = int(hf_cfg.get("bos_token_id", 128000))
    eos_id = int(hf_cfg.get("eos_token_id", 128001))
    pad_id = int(hf_cfg.get("pad_token_id") if hf_cfg.get("pad_token_id") is not None else eos_id)
    if pad_id < 0 or pad_id >= vocab_size:
        pad_id = 0

    trace_cfg: dict[str, Any] = {
        "vocab_size": vocab_size,
        "pad_token_id": pad_id,
        "mask_token_id": pad_id,
        "bos_token_id": bos_id,
        "eos_token_id": eos_id,
        "max_seq_len": max_seq_len,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "d_ff": d_ff,
        "eps": eps,
        "kv_lora_rank": 0,
        "qk_nope_head_dim": 0,
        "qk_rope_head_dim": head_dim,
        "v_head_dim": head_dim,
        "tie_word_embeddings": tie_emb,
        "attn_type": "gqa",
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "rope_theta": rope_theta,
        "rope_interleaved": False,
        "use_gated_ffn": True,
    }

    reader = SafeTensorsReader(src_dir)
    tensor_index: list[dict[str, Any]] = []
    resident_bytes = io.BytesIO()
    pages_bytes = io.BytesIO()
    current_page_id = 0

    # 1. Embeddings residentes (compactables a float16 / int8) y norma final
    tok_emb = reader.get_float32("model.embed_tokens.weight")
    _write_resident_tensor("tok_emb.weight", tok_emb, resident_bytes, tensor_index, dtype=embedding_dtype)
    del tok_emb

    if not tie_emb and reader.has_tensor("lm_head.weight"):
        lm_head = reader.get_float32("lm_head.weight")
        _write_resident_tensor("lm_head.weight", lm_head, resident_bytes, tensor_index, dtype=embedding_dtype)
        del lm_head

    final_norm = reader.get_float32("model.norm.weight")
    _write_resident_tensor("final_norm.weight", final_norm, resident_bytes, tensor_index, dtype="float32")
    del final_norm

    # 2. Capas Transformer GQA + ReLU² gated (streaming capa a capa)
    proj_map = [
        ("self_attn.q_proj.weight", "attn.w_q"),
        ("self_attn.k_proj.weight", "attn.w_k"),
        ("self_attn.v_proj.weight", "attn.w_v"),
        ("self_attn.o_proj.weight", "attn.w_out"),
        ("mlp.gate_proj.weight", "ffn.gate"),
        ("mlp.up_proj.weight", "ffn.up"),
        ("mlp.down_proj.weight", "ffn.down"),
    ]
    norm_map = [
        ("input_layernorm.weight", "attn_norm.weight"),
        ("post_attention_layernorm.weight", "ffn_norm.weight"),
        ("self_attn.attn_sub_norm.weight", "attn.w_out.sub_ln.weight"),
        ("mlp.ffn_sub_norm.weight", "ffn.down.sub_ln.weight"),
    ]

    for i in range(n_layers):
        prefix = f"model.layers.{i}"
        t_prefix = f"layers.{i}"

        for hf_suffix, trace_suffix in norm_map:
            hf_key = f"{prefix}.{hf_suffix}"
            if reader.has_tensor(hf_key):
                arr = reader.get_float32(hf_key)
                _write_resident_tensor(f"{t_prefix}.{trace_suffix}", arr, resident_bytes, tensor_index, dtype="float32")

        for hf_suffix, trace_suffix in proj_map:
            hf_key = f"{prefix}.{hf_suffix}"
            codes, scale_val = _load_ternary_projection(
                reader, hf_key, pack_layout=pack_layout, invert_scale=invert_scale
            )
            current_page_id = _write_paged_projection(
                name=f"{t_prefix}.{trace_suffix}",
                codes_np=codes,
                scale_val=scale_val,
                page_size=page_size,
                pages_bytes=pages_bytes,
                tensor_index=tensor_index,
                current_page_id=current_page_id,
            )

    # 3. Ensamblar archivo binario Paged BitCOS v1.1
    idx_payload = {
        "tensors": tensor_index,
        "format": "fast",
        "architecture": "trace-lm-v1",
        "config": trace_cfg,
    }
    idx_json = json.dumps(idx_payload).encode("utf-8")
    exp_json = json.dumps({"experts": {}}).encode("utf-8")

    header_size = 64
    idx_offset = header_size
    exp_offset = idx_offset + len(idx_json)
    res_offset = exp_offset + len(exp_json)
    res_len = resident_bytes.tell()
    raw_pages_offset = res_offset + res_len
    pages_offset = ((raw_pages_offset + page_size - 1) // page_size) * page_size
    padding_to_pages = bytes(pages_offset - raw_pages_offset)

    header_blob = struct.pack(
        "<8sIIIQQQQQ",
        MAGIC_PBITCOS,
        page_size,
        1,  # fmt_code = 1 (fast)
        current_page_id,
        idx_offset,
        exp_offset,
        res_offset,
        pages_offset,
        pages_offset + pages_bytes.tell(),
    )
    if len(header_blob) < header_size:
        header_blob += bytes(header_size - len(header_blob))

    out_path = Path(output_bin)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(header_blob)
        f.write(idx_json)
        f.write(exp_json)
        f.write(resident_bytes.getvalue())
        f.write(padding_to_pages)
        f.write(pages_bytes.getvalue())

    return {
        "output_bin": str(out_path),
        "total_bytes": os.path.getsize(out_path),
        "total_pages": current_page_id,
        "embedding_dtype": embedding_dtype,
        "n_layers": n_layers,
        "vocab_size": vocab_size,
    }


# =====================================================================
# Conversor HuggingFace tokenizer.json (LLaMA-3 Byte-Level BPE) -> TRTK v2
# =====================================================================
def _gpt2_unicode_to_bytes() -> dict[str, int]:
    """Tabla inversa del mapeo ByteLevel de GPT-2 / LLaMA-3."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def _decode_bytelevel_piece(piece_str: str, u2b: dict[str, int]) -> bytes:
    out = bytearray()
    for ch in piece_str:
        b = u2b.get(ch)
        if b is not None:
            out.append(b)
        else:
            out.extend(ch.encode("utf-8"))
    return bytes(out)


def export_hf_tokenizer_to_trtk_v2(
    tokenizer_json_path: str | Path,
    output_trtk_path: str | Path,
) -> dict[str, Any]:
    """Convierte un `tokenizer.json` Byte-Level BPE (LLaMA-3 128K) a binario nativo TRTK v2."""
    with open(tokenizer_json_path, encoding="utf-8") as f:
        tj = json.load(f)

    model_section = tj.get("model", {})
    vocab_map: dict[str, int] = {k: int(v) for k, v in model_section.get("vocab", {}).items()}
    added_tokens = tj.get("added_tokens", [])
    special_by_str: dict[str, int] = {}
    for item in added_tokens:
        tid = int(item["id"])
        content = str(item["content"])
        vocab_map[content] = tid
        special_by_str[content] = tid

    if not vocab_map:
        raise ValueError(f"Vocabulario vacío en {tokenizer_json_path}")

    vocab_size = max(vocab_map.values()) + 1
    u2b = _gpt2_unicode_to_bytes()

    # 1. Decodificar piezas por ID y construir byte_to_id[256]
    pieces: list[bytes] = [b""] * vocab_size
    byte_to_id = [0] * 256
    byte_found = [False] * 256
    added_id_set = set(special_by_str.values())

    for tok_str, tid in vocab_map.items():
        if tid in added_id_set:
            raw = tok_str.encode("utf-8")
        else:
            raw = _decode_bytelevel_piece(tok_str, u2b)
        pieces[tid] = raw
        if tid not in added_id_set and len(raw) == 1 and not byte_found[raw[0]]:
            byte_to_id[raw[0]] = tid
            byte_found[raw[0]] = True

    # 2. Resolver merges (soporta formato "a b" y ["a", "b"])
    raw_merges = model_section.get("merges", [])
    merges_v2: list[tuple[int, int, int]] = []
    for entry in raw_merges:
        if isinstance(entry, str):
            parts = entry.split(" ")
            if len(parts) != 2:
                continue
            left_s, right_s = parts[0], parts[1]
        elif isinstance(entry, list) and len(entry) == 2:
            left_s, right_s = str(entry[0]), str(entry[1])
        else:
            continue
        lid = vocab_map.get(left_s)
        rid = vocab_map.get(right_s)
        res_id = vocab_map.get(left_s + right_s)
        if lid is not None and rid is not None and res_id is not None:
            merges_v2.append((lid, rid, res_id))

    # 3. Detectar IDs especiales LLaMA-3
    invalid_u32 = 0xFFFFFFFF
    bos_id = special_by_str.get("<|begin_of_text|>", special_by_str.get("<s>", 128000 if vocab_size > 128000 else 0))
    eos_id = special_by_str.get("<|end_of_text|>", special_by_str.get("</s>", 128001 if vocab_size > 128001 else 0))
    pad_id = special_by_str.get("<|finetune_right_pad_id|>", eos_id)
    mask_id = pad_id
    start_header_id = special_by_str.get("<|start_header_id|>", invalid_u32)
    end_header_id = special_by_str.get("<|end_header_id|>", invalid_u32)
    eot_id = special_by_str.get("<|eot_id|>", invalid_u32)

    out_path = Path(output_trtk_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        # Header de 32 bytes (version = 2)
        f.write(struct.pack("<8I", TRTK_MAGIC, 2, vocab_size, len(merges_v2), pad_id, mask_id, bos_id, eos_id))
        # byte_to_id[256] (1024 bytes)
        f.write(struct.pack("<256I", *byte_to_id))
        # special_ids[4] (16 bytes)
        f.write(struct.pack("<4I", start_header_id, end_header_id, eot_id, 0))
        # merges_v2 (12 bytes c/u)
        for lid, rid, res_id in merges_v2:
            f.write(struct.pack("<3I", lid, rid, res_id))
        # pieces (uint16 len + raw bytes)
        for raw in pieces:
            if len(raw) > 65535:
                raw = raw[:65535]
            f.write(struct.pack("<H", len(raw)))
            if raw:
                f.write(raw)

    return {
        "output_trtk": str(out_path),
        "vocab_size": vocab_size,
        "num_merges": len(merges_v2),
        "bos_id": bos_id,
        "eos_id": eos_id,
        "eot_id": eot_id if eot_id != invalid_u32 else -1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Importa un checkpoint HuggingFace BitNet b1.58 (GQA) a Paged BitCOS v1.1 + TRTK v2."
    )
    parser.add_argument("--model-dir", required=True, help="Directorio local con config.json y model.safetensors")
    parser.add_argument("--output-model", required=True, help="Ruta de salida para el binario Paged BitCOS (.bin)")
    parser.add_argument("--output-tokenizer", default=None, help="Ruta de salida opcional para TRTK v2 (.bin)")
    parser.add_argument(
        "--embedding-dtype",
        choices=("float32", "float16", "int8"),
        default="float16",
        help="Precisión de embeddings residentes (default: float16, ~657 MB en 2B-4T)",
    )
    parser.add_argument(
        "--pack-layout",
        choices=("hf_chunk", "interleaved"),
        default="hf_chunk",
        help="Layout de empaquetado uint8 para pesos BitNet empaquetados (default: hf_chunk)",
    )
    parser.add_argument("--invert-scale", action="store_true", help="Invierte weight_scale (1 / scale) si el checkpoint lo requiere")
    args = parser.parse_args()

    model_info = import_bitnet_model(
        model_dir=args.model_dir,
        output_bin=args.output_model,
        embedding_dtype=args.embedding_dtype,
        pack_layout=args.pack_layout,
        invert_scale=args.invert_scale,
    )
    print(f"[OK] Modelo exportado: {model_info}")

    if args.output_tokenizer:
        tok_json = Path(args.model_dir) / "tokenizer.json"
        if not tok_json.exists():
            raise FileNotFoundError(f"No se encontró tokenizer.json en {args.model_dir}")
        tok_info = export_hf_tokenizer_to_trtk_v2(tok_json, args.output_tokenizer)
        print(f"[OK] Tokenizador TRTK v2 exportado: {tok_info}")


if __name__ == "__main__":
    main()
