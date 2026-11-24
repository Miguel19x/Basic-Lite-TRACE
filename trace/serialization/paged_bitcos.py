# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available

import dataclasses
import io
import json
import mmap
import os
import struct
from trace.layers.bitlinear import BitLinear
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# Magic identificador para el formato de ejecución por páginas (Paged BitCOS v1)
MAGIC_PBITCOS = b"PBITCOS1"
DEFAULT_PAGE_SIZE = 4096  # 4 KB por página, óptimo para paginación de SO y lecturas alineadas SSD


def count_fp32_weights_in_bitlinear(model: nn.Module) -> int:
    """Cuenta elementos FP32 en los parámetros y cachés de BitLinear."""
    total_fp32 = 0
    for _, module in model.named_modules():
        if isinstance(module, BitLinear):
            if module.weight is not None and module.weight.dtype == torch.float32 and module.weight.numel() > 0:
                total_fp32 += module.weight.numel()
            if hasattr(module, "cached_w_eff") and module.cached_w_eff is not None:
                if module.cached_w_eff.numel() > 0 and module.cached_w_eff.dtype == torch.float32:
                    total_fp32 += module.cached_w_eff.numel()
    return total_fp32


class PagedBitCOSSerializer:
    """
    Serializador y motor de paginación para pesos ternarios.
    Estructura del archivo:
      [0..64]   Cabecera principal (Magic, versión, page_size, offsets y metadatos)
      [Offset]  Índice de tensores (nombres, shapes, escalas, lista de páginas)
      [Offset]  Índice de expertos (mapeo expert_id -> páginas necesarias)
      [Offset]  Embeddings y normalizaciones (pesos pequeños 100% residentes en RAM)
      [Offset]  Páginas de matrices ternarias alineadas a page_size
    """

    @staticmethod
    def pack_ternary_page(
        row_start: int,
        out_features: int,
        in_features: int,
        scale_w: float,
        q_w_chunk: np.ndarray,
        page_size: int = DEFAULT_PAGE_SIZE,
        format_type: str = "compact",
    ) -> bytes:
        """
        Empaqueta un bloque de filas de la matriz ternaria en una página fija.
        - compact: presencia (1 bit) + signos de no-ceros (1 bit).
        - fast: 2 bits directos por peso {-1: 00, 0: 01, +1: 10} optimizado para decodificación veloz.
        """
        rows, cols = q_w_chunk.shape
        flat = q_w_chunk.flatten().astype(np.int8)

        if format_type == "compact":
            presence_bits = (flat != 0).astype(np.uint8)
            nonzero_vals = flat[flat != 0]
            sign_bits = (nonzero_vals > 0).astype(np.uint8)

            packed_presence = np.packbits(presence_bits).tobytes()
            packed_signs = np.packbits(sign_bits).tobytes()
            payload = packed_presence + packed_signs
            fmt_code = 0
            len_p1 = len(packed_presence)
            len_p2 = len(packed_signs)
        else:  # 'fast'
            # Mapeo: -1 -> 0, 0 -> 1, +1 -> 2 (cabe en 2 bits)
            # Pauta técnica Fase 2: Alineación estricta por fila a múltiplos de 32 pesos (8 bytes = 1 vector AVX2)
            padded_cols = ((cols + 31) // 32) * 32
            if padded_cols > cols:
                pad_width = padded_cols - cols
                q_w_padded = np.pad(q_w_chunk, ((0, 0), (0, pad_width)), mode="constant", constant_values=0)
            else:
                q_w_padded = q_w_chunk

            flat_padded = q_w_padded.flatten().astype(np.int8)
            codes = (flat_padded + 1).astype(np.uint8) & 0x03
            b0 = codes[0::4]
            b1 = codes[1::4] << 2
            b2 = codes[2::4] << 4
            b3 = codes[3::4] << 6
            packed = (b0 | b1 | b2 | b3).tobytes()
            payload = packed
            fmt_code = 1
            len_p1 = len(packed)
            len_p2 = 0

        header_fmt = "<IIHHfII"
        header_bytes = struct.pack(
            header_fmt,
            row_start,
            rows,
            cols,
            fmt_code,
            float(scale_w),
            len_p1,
            len_p2,
        )
        page_data = header_bytes + payload
        if len(page_data) > page_size:
            raise ValueError(
                f"El bloque ({rows}x{cols}) supera el tamaño de página {page_size} bytes ({len(page_data)} bytes)"
            )
        # Relleno hasta el límite exacto de la página
        padding = bytes(page_size - len(page_data))
        return page_data + padding

    @classmethod
    def export_paged_model(
        cls,
        model: nn.Module,
        output_path: str,
        page_size: int = DEFAULT_PAGE_SIZE,
        format_type: str = "compact",
        architecture: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Exporta el modelo a un archivo binario Paged BitCOS alineado a páginas.
        Separa componentes residentes (embeddings, normas) de páginas ternarias (pesos de proyección y expertos).
        """
        if architecture is None and config is None and hasattr(model, "cfg"):
            cfg_obj = model.cfg
            cls_name = cfg_obj.__class__.__name__
            if cls_name == "TRACELMConfig":
                architecture = "trace-lm-v1"
                if dataclasses.is_dataclass(cfg_obj):
                    config = dataclasses.asdict(cfg_obj)
            elif cls_name == "HybridConfig":
                architecture = "trace-causal-v1"
                if dataclasses.is_dataclass(cfg_obj):
                    config = dataclasses.asdict(cfg_obj)

        tensor_index: list[dict[str, Any]] = []
        expert_index: dict[int, list[int]] = {}
        resident_bytes = io.BytesIO()
        pages_bytes = io.BytesIO()

        current_page_id = 0
        named_modules = dict(model.named_modules())

        for name, module in named_modules.items():
            if isinstance(module, BitLinear):
                # Extraer pesos cuantizados y escala
                with torch.no_grad():
                    _, q_w, scale_w = module.quantize_weights()
                    codes_np = q_w.cpu().numpy().astype(np.int8)
                    scale_val = float(scale_w.item())

                out_f, in_f = codes_np.shape
                # Calcular cuántas filas caben por página de page_size
                if format_type == "fast":
                    padded_in_f = ((in_f + 31) // 32) * 32
                    row_bytes = padded_in_f // 4
                    rows_per_page = max(1, (page_size - 64) // row_bytes)
                else:
                    # compact: cada peso ocupa ~1.58 bits, redondeamos con margen seguro
                    rows_per_page = max(1, (page_size - 64) // max(1, int(in_f * 0.25) + 4))

                tensor_pages = []
                for r_start in range(0, out_f, rows_per_page):
                    r_end = min(out_f, r_start + rows_per_page)
                    chunk = codes_np[r_start:r_end, :]
                    page_blob = cls.pack_ternary_page(
                        row_start=r_start,
                        out_features=out_f,
                        in_features=in_f,
                        scale_w=scale_val,
                        q_w_chunk=chunk,
                        page_size=page_size,
                        format_type=format_type,
                    )
                    pages_bytes.write(page_blob)
                    tensor_pages.append(current_page_id)
                    current_page_id += 1

                is_expert = "expert" in name
                if is_expert:
                    parts = name.split(".")
                    exp_id = -1
                    for p in parts:
                        if p.isdigit():
                            exp_id = int(p)
                            break
                        elif p.startswith("expert_") or p == "shared_expert":
                            exp_id = 999 if p == "shared_expert" else -1
                    if exp_id != -1:
                        expert_index.setdefault(exp_id, []).extend(tensor_pages)

                tensor_index.append({
                    "name": name,
                    "shape": [out_f, in_f],
                    "is_resident": False,
                    "scale": scale_val,
                    "num_pages": len(tensor_pages),
                    "page_ids": tensor_pages,
                })

        # Serializar parámetros residentes (normas, embeddings, gates)
        for name, param in model.named_parameters():
            # Si no es peso de BitLinear
            is_bitlinear_weight = any(
                name == f"{mod_name}.weight"
                for mod_name, mod in named_modules.items()
                if isinstance(mod, BitLinear)
            )
            if not is_bitlinear_weight and param.numel() > 0:
                p_np = param.detach().cpu().numpy().astype(np.float32)
                p_bytes = p_np.tobytes()
                p_offset = resident_bytes.tell()
                resident_bytes.write(p_bytes)
                tensor_index.append({
                    "name": name,
                    "shape": list(p_np.shape),
                    "is_resident": True,
                    "offset": p_offset,
                    "length": len(p_bytes),
                    "dtype": "float32",
                })

        # Serializar índices en bloques JSON/Binario con metadatos de arquitectura
        idx_payload: dict[str, Any] = {
            "tensors": tensor_index,
            "format": format_type,
        }
        if architecture is not None:
            idx_payload["architecture"] = architecture
        if config is not None:
            idx_payload["config"] = config

        idx_json = json.dumps(idx_payload).encode()
        exp_json = json.dumps({"experts": expert_index}).encode()

        # Construir archivo final
        header_size = 64
        idx_offset = header_size
        idx_len = len(idx_json)
        exp_offset = idx_offset + idx_len
        exp_len = len(exp_json)
        res_offset = exp_offset + exp_len
        res_len = resident_bytes.tell()
        # Alinear pages_offset al tamaño de página
        raw_pages_offset = res_offset + res_len
        pages_offset = ((raw_pages_offset + page_size - 1) // page_size) * page_size
        padding_to_pages = bytes(pages_offset - raw_pages_offset)

        total_pages = current_page_id
        fmt_code = 0 if format_type == "compact" else 1

        header_blob = struct.pack(
            "<8sIIIQQQQQ",
            MAGIC_PBITCOS,
            page_size,
            fmt_code,
            total_pages,
            idx_offset,
            exp_offset,
            res_offset,
            pages_offset,
            pages_offset + pages_bytes.tell(),
        )
        if len(header_blob) < header_size:
            header_blob += bytes(header_size - len(header_blob))

        with open(output_path, "wb") as f:
            f.write(header_blob)
            f.write(idx_json)
            f.write(exp_json)
            f.write(resident_bytes.getvalue())
            f.write(padding_to_pages)
            f.write(pages_bytes.getvalue())

        total_file_size = os.path.getsize(output_path)
        return {
            "path": output_path,
            "total_pages": total_pages,
            "page_size": page_size,
            "format": format_type,
            "total_bytes": total_file_size,
            "tensors_count": len(tensor_index),
            "experts_count": len(expert_index),
            "architecture": architecture,
        }


class PagedBitCOSReader:
    """
    Lector de alto rendimiento para Paged BitCOS.
    Permite acceso por demanda a páginas individuales vía mmap o seek sin cargar el modelo completo en RAM.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._f = open(file_path, "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        self._parse_header()

    def _parse_header(self) -> None:
        header_size = struct.calcsize("<8sIIIQQQQQ")
        magic, page_sz, fmt_code, total_pgs, idx_off, exp_off, res_off, pgs_off, total_sz = (
            struct.unpack("<8sIIIQQQQQ", self._mm[:header_size])
        )
        assert magic == MAGIC_PBITCOS, f"Magic inválido: {magic}"
        self.page_size = page_sz
        self.format_type = "compact" if fmt_code == 0 else "fast"
        self.total_pages = total_pgs
        self.idx_offset = idx_off
        self.exp_offset = exp_off
        self.res_offset = res_off
        self.pages_offset = pgs_off
        self.total_size = total_sz

        idx_data = self._mm[idx_off:exp_off].decode("utf-8")
        self.index_data = json.loads(idx_data)
        self.tensors_map = {t["name"]: t for t in self.index_data["tensors"]}

        exp_data = self._mm[exp_off:res_off].decode("utf-8")
        self.expert_map = json.loads(exp_data)["experts"]

    def get_architecture(self) -> str | None:
        """Devuelve el identificador de arquitectura o None si no existe en el binario."""
        return self.index_data.get("architecture")

    def get_model_config(self) -> dict[str, Any] | None:
        """Devuelve el diccionario de configuración del modelo o None si no existe."""
        return self.index_data.get("config")

    def read_page_bytes(self, page_id: int) -> bytes:
        """Lee una sola página de 4KB desde el archivo mapeado en memoria (cero sobrecarga de descompresión global)."""
        if page_id >= self.total_pages:
            raise IndexError(f"Page ID {page_id} fuera de rango ({self.total_pages})")
        start = self.pages_offset + page_id * self.page_size
        return self._mm[start : start + self.page_size]

    def read_resident_tensor(self, name: str) -> np.ndarray | None:
        """Recupera un tensor residente (FP32) directamente del archivo."""
        info = self.tensors_map.get(name)
        if not info or not info.get("is_resident", False):
            return None
        start = self.res_offset + info["offset"]
        length = info["length"]
        raw = self._mm[start : start + length]
        arr = np.frombuffer(raw, dtype=np.float32).reshape(info["shape"])
        return arr.copy()

    def get_expert_pages(self, expert_id: int) -> list[int]:
        """Retorna las páginas que componen a un experto determinado."""
        return self.expert_map.get(str(expert_id), [])

    def load_into_model(self, model: nn.Module) -> int:
        """
        Carga íntegramente los parámetros residentes y las páginas ternarias
        desde el archivo Paged BitCOS hacia una instancia de modelo vacía/limpia.
        """
        from trace.layers.bitlinear import BitLinear
        from trace.ops.paged_kernel import unpack_page_to_dense

        # 1. Cargar tensores residentes (embeddings, normas, etc.)
        loaded_resident = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                arr = self.read_resident_tensor(name)
                if arr is not None:
                    param.data.copy_(torch.from_numpy(arr))
                    loaded_resident += 1

        # Preservar tied embeddings explícitamente si aplica
        if (
            hasattr(model, "lm_head")
            and hasattr(model, "tok_emb")
            and getattr(getattr(model, "cfg", None), "tie_word_embeddings", False)
        ):
            model.lm_head.weight = model.tok_emb.weight

        # 2. Cargar capas ternarias BitLinear desde las páginas
        loaded_bitlinear = 0
        named_modules = dict(model.named_modules())
        for name, module in named_modules.items():
            if isinstance(module, BitLinear):
                info = self.tensors_map.get(name)
                if info and not info.get("is_resident", False):
                    out_f, in_f = info["shape"]
                    full_codes = torch.zeros(out_f, in_f, dtype=torch.int8)

                    scales = []
                    for pid in info["page_ids"]:
                        page_bytes = self.read_page_bytes(pid)
                        r_s, r_cnt, _, sc, codes = unpack_page_to_dense(page_bytes)
                        full_codes[r_s : r_s + r_cnt, :] = codes
                        scales.append(sc)

                    module.int8_codes.copy_(full_codes)
                    # En capas multipágina, scales[0] garantiza la escala original exacta sin redondeo residual
                    scale_val = scales[0] if scales else info["scale"]
                    module.packed_scale.fill_(scale_val)
                    # Criterio Fase 2/3: 0 pesos FP32 en RAM para proyecciones BitLinear
                    module.cached_w_eff = torch.empty(0)
                    module.weight = nn.Parameter(torch.empty(0), requires_grad=False)
                    module.use_loaded_bitcos = True
                    loaded_bitlinear += 1

        return loaded_resident + loaded_bitlinear

    def get_accounting_breakdown(self) -> dict[str, Any]:
        """
        Desglosa con precisión contable cada sección del archivo binario Paged BitCOS.
        Aclara la discrepancia entre el tamaño de páginas ternarias y el tamaño total del binario.
        """
        header_len = self.idx_offset
        idx_len = self.exp_offset - self.idx_offset
        exp_len = self.res_offset - self.exp_offset
        pages_len = self.total_size - self.pages_offset

        # Desglose de parámetros residentes en FP32
        res_params_count = sum(1 for t in self.index_data["tensors"] if t.get("is_resident", False))
        res_params_bytes = sum(t["length"] for t in self.index_data["tensors"] if t.get("is_resident", False))
        padding_bytes = (self.pages_offset - self.res_offset) - res_params_bytes

        return {
            "total_file_bytes": self.total_size,
            "header_bytes": header_len,
            "tensor_index_json_bytes": idx_len,
            "expert_index_json_bytes": exp_len,
            "resident_fp32_params_count": res_params_count,
            "resident_fp32_params_bytes": res_params_bytes,
            "alignment_padding_bytes": padding_bytes,
            "ternary_pages_count": self.total_pages,
            "ternary_pages_bytes": pages_len,
            "page_size": self.page_size,
            "format": self.format_type,
            "budget_30b_gb": {
                "ideal_ternary_log2_3_bits_gb": 5.94,
                "current_fast_2bit_lut_gb": round(30.0 * 2.0 / 8.0, 2),
            },
        }

    def close(self) -> None:
        if hasattr(self, "_mm") and self._mm is not None:
            self._mm.close()
        if hasattr(self, "_f") and self._f is not None:
            self._f.close()

    def __del__(self) -> None:
        self.close()
