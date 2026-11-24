# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available

import random
import re

import torch
from torch.utils.data import Dataset

from config import HybridConfig


class ProceduralCompositionalDataset(Dataset):
    """
    Universo exacto de 2880 combinaciones:
      10 nodos x 8 perfiles x 6 redundancias x 6 cargas/estrés = 2880 combinaciones.
    Formato de salida estricto de ancho fijo (3 dígitos con ceros a la izquierda):
      watt: 087\n
      ms: 045\n
    Particiones disjuntas pre-registradas (§3.2):
      - test-composicional: 480 combinaciones (8 pares perfil-carga reservados x 10 nodos x 6 redundancias)
      - test-azar: 240 combinaciones (10% del remanente IID)
      - dev: 108 combinaciones (~4.5% del remanente)
      - train: 2052 combinaciones
    """
    NODOS = [
        ("sensor", 10, 4),
        ("micro", 15, 6),
        ("gateway", 28, 9),
        ("edge_box", 45, 12),
        ("switch", 55, 3),
        ("router", 90, 14),
        ("core_switch", 120, 8),
        ("server", 140, 25),
        ("cluster", 210, 40),
        ("datacenter", 280, 50),
    ]

    CARGAS = [
        ("idle", 0.4, 1.0),
        ("baja", 0.7, 1.2),
        ("media", 1.0, 1.5),
        ("alta", 1.4, 2.0),
        ("pico", 1.8, 2.7),
        ("saturada", 2.2, 3.5),
    ]

    REDUNDANCIAS = [
        ("ninguna", 0, 0),
        ("simple", 8, 2),
        ("doble", 18, 5),
        ("triple", 25, 7),
        ("mesh", 35, 10),
        ("ring", 45, 14),
    ]

    PERFILES = [
        ("eco", -10, 5),
        ("base", 0, 0),
        ("backup", 5, 12),
        ("edge", 8, -1),
        ("boost", 12, -2),
        ("batch", 15, 6),
        ("secure", 20, 8),
        ("turbo", 25, -4),
    ]

    # 8 pares cruzados (perfil, carga) reservados para test-composicional
    # 8 pares x 10 nodos x 6 redundancias = exactamente 480 combinaciones excluidas de train
    HELD_OUT_PAIRS = {
        ("eco", "alta"),
        ("base", "pico"),
        ("backup", "idle"),
        ("edge", "saturada"),
        ("boost", "media"),
        ("batch", "baja"),
        ("secure", "alta"),
        ("turbo", "saturada"),
    }

    def __init__(self, cfg: HybridConfig, split: str = "train", num_samples: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.split = split

        all_records = []
        comp_pool = []
        remaining_pool = []

        self.record_meta: dict[tuple[bytes, bytes], tuple[int, int, int, int]] = {}

        for n_name, b_w, b_ms in self.NODOS:
            for c_name, m_w, m_ms in self.CARGAS:
                for r_name, a_w, a_ms in self.REDUNDANCIAS:
                    for p_name, o_w, o_ms in self.PERFILES:
                        sub_w = int(round(b_w * m_w))
                        sub_ms = int(round(b_ms * m_ms))
                        watt = max(5, min(999, sub_w + a_w + o_w))
                        ms = max(1, min(999, sub_ms + a_ms + o_ms))

                        if getattr(cfg, "use_scratchpad", False):
                            sub_w = max(1, min(999, sub_w))
                            sub_ms = max(1, min(999, sub_ms))
                            target_str = f"pw: {sub_w:03d}\npms: {sub_ms:03d}\nwatt: {watt:03d}\nms: {ms:03d}\n"
                        else:
                            target_str = f"watt: {watt:03d}\nms: {ms:03d}\n"

                        if getattr(cfg, "use_explicit_operands", False):
                            prompt_str = (
                                f"nodo: {n_name}\ncarga: {c_name}\nred: {r_name}\nperfil: {p_name}\n"
                                f"bw: {b_w} mw: {m_w} aw: {a_w} ow: {o_w} bms: {b_ms} mms: {m_ms} ams: {a_ms} oms: {o_ms}\n"
                            )
                        else:
                            prompt_str = f"nodo: {n_name}\ncarga: {c_name}\nred: {r_name}\nperfil: {p_name}\n"

                        p_bytes = prompt_str.encode("utf-8")
                        t_bytes = target_str.encode("utf-8")
                        record = (p_bytes, t_bytes, watt, ms)
                        self.record_meta[(p_bytes, t_bytes)] = (sub_w, sub_ms, watt, ms)
                        all_records.append(record)

                        if (p_name, c_name) in self.HELD_OUT_PAIRS:
                            comp_pool.append(record)
                        else:
                            remaining_pool.append(record)

        assert len(all_records) == 2880, f"Error en espacio combinatorio: {len(all_records)} != 2880"
        assert len(comp_pool) == 480, f"Error en test composicional: {len(comp_pool)} != 480"
        assert len(remaining_pool) == 2400, f"Error en pool remanente: {len(remaining_pool)} != 2400"

        # Particionar los 2400 restantes de forma reproducible con semilla 42
        rng_part = random.Random(42)
        shuffled_rem = list(remaining_pool)
        rng_part.shuffle(shuffled_rem)

        # 240 test-azar (10%), 108 dev (~4.5%), 2052 train
        test_azar_pool = shuffled_rem[:240]
        dev_pool = shuffled_rem[240:348]
        train_pool = shuffled_rem[348:]

        assert len(test_azar_pool) == 240
        assert len(dev_pool) == 108
        assert len(train_pool) == 2052

        self.pools = {
            "all": all_records,
            "train": train_pool,
            "dev": dev_pool,
            "test-azar": test_azar_pool,
            "test-composicional": comp_pool,
        }

        if split not in self.pools:
            raise ValueError(f"Split desconocido '{split}'. Opciones válidas: {sorted(self.pools.keys())}")
        chosen_pool = self.pools[split]
        self.unique_records = list(chosen_pool)

        if num_samples is None:
            self.samples = list(chosen_pool)
        else:
            rng_sample = random.Random(cfg.seed)
            if num_samples <= len(chosen_pool):
                self.samples = rng_sample.sample(chosen_pool, num_samples)
            else:
                self.samples = [rng_sample.choice(chosen_pool) for _ in range(num_samples)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_b, target_b, _, _ = self.samples[idx]
        raw_len = 1 + len(prompt_b) + len(target_b) + 1
        assert raw_len <= self.cfg.max_seq_len, (
            f"Error de longitud: {raw_len} > {self.cfg.max_seq_len} bytes."
        )

        seq = [self.cfg.bos_token_id] + list(prompt_b) + list(target_b) + [self.cfg.eos_token_id]
        prompt_len = 1 + len(prompt_b)

        seq = seq[: self.cfg.max_seq_len]
        pad_len = self.cfg.max_seq_len - len(seq)
        seq = seq + [self.cfg.pad_token_id] * pad_len

        x_0 = torch.tensor(seq, dtype=torch.long)
        can_mask = torch.zeros(self.cfg.max_seq_len, dtype=torch.bool)
        can_mask[prompt_len:] = True

        digit_mask = torch.zeros(self.cfg.max_seq_len, dtype=torch.bool)
        target_end = min(self.cfg.max_seq_len, prompt_len + len(target_b))
        for pos in range(prompt_len, target_end):
            if 48 <= seq[pos] <= 57:
                digit_mask[pos] = True

        return x_0, can_mask, digit_mask


_WATT_MS_FIXED_RE = re.compile(r"watt:\s*(\d{3})\s*\nms:\s*(\d{3})")
_SCRATCHPAD_FIXED_RE = re.compile(r"pw:\s*(\d{3})\s*\npms:\s*(\d{3})\s*\nwatt:\s*(\d{3})\s*\nms:\s*(\d{3})")


def parse_fixed_prediction(text: str) -> tuple[bool, int | None, int | None]:
    """Analizador estricto para salidas de ancho fijo (3 dígitos obligatorios)."""
    m = _WATT_MS_FIXED_RE.fullmatch(text.strip())
    if m:
        try:
            return True, int(m.group(1)), int(m.group(2))
        except ValueError:
            pass
    return False, None, None


def parse_fixed_scratchpad(text: str) -> tuple[bool, int | None, int | None, int | None, int | None]:
    """Analizador estricto para salidas completas con scratchpad de ancho fijo."""
    m = _SCRATCHPAD_FIXED_RE.fullmatch(text.strip())
    if m:
        try:
            return True, int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        except ValueError:
            pass
    return False, None, None, None, None

