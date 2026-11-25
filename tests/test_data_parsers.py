# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Tests unitarios y de propiedades para parsers de ancho fijo y particiones de data.py."""

import random

import pytest

from config import HybridConfig
from data import (
    ProceduralCompositionalDataset,
    parse_fixed_prediction,
    parse_fixed_scratchpad,
)


@pytest.mark.parametrize("watt,ms", [(0, 0), (5, 1), (87, 45), (999, 999)])
def test_parse_fixed_prediction_roundtrip_boundaries(watt: int, ms: int):
    text = f"watt: {watt:03d}\nms: {ms:03d}\n"
    assert parse_fixed_prediction(text) == (True, watt, ms)


def test_parse_fixed_prediction_roundtrip_property_sweep():
    rng = random.Random(2026)
    for _ in range(50):
        w, m = rng.randint(0, 999), rng.randint(0, 999)
        assert parse_fixed_prediction(f"watt: {w:03d}\nms: {m:03d}\n") == (True, w, m)


@pytest.mark.parametrize(
    "bad_text",
    [
        "",
        "watt: 87\nms: 045\n",
        "watt: 087\nms: 45\n",
        "watt: 1000\nms: 045\n",
        "watt: abc\nms: 045\n",
        "watt: 087\n",
        "watt: 087\nms: 045\nextra: 001\n",
    ],
)
def test_parse_fixed_prediction_rejects_malformed_inputs(bad_text: str):
    assert parse_fixed_prediction(bad_text) == (False, None, None)


@pytest.mark.parametrize(
    "pw,pms,watt,ms",
    [(0, 0, 0, 0), (1, 1, 5, 1), (120, 45, 155, 55), (999, 999, 999, 999)],
)
def test_parse_fixed_scratchpad_roundtrip_boundaries(pw: int, pms: int, watt: int, ms: int):
    text = f"pw: {pw:03d}\npms: {pms:03d}\nwatt: {watt:03d}\nms: {ms:03d}\n"
    assert parse_fixed_scratchpad(text) == (True, pw, pms, watt, ms)


@pytest.mark.parametrize(
    "bad_text",
    [
        "",
        "watt: 087\nms: 045\n",
        "pw: 12\npms: 045\nwatt: 087\nms: 045\n",
        "pw: 012\npms: 045\nwatt: 087\nms: 1000\n",
        "pw: ???\npms: ???\nwatt: ???\nms: ???\n",
    ],
)
def test_parse_fixed_scratchpad_rejects_malformed_inputs(bad_text: str):
    assert parse_fixed_scratchpad(bad_text) == (False, None, None, None, None)


def test_dataset_splits_are_pairwise_disjoint():
    cfg = HybridConfig()
    splits = {
        name: set(ProceduralCompositionalDataset(cfg, split=name).samples)
        for name in ("train", "dev", "test-azar", "test-composicional")
    }
    assert len(splits["train"]) == 2052
    assert len(splits["dev"]) == 108
    assert len(splits["test-azar"]) == 240
    assert len(splits["test-composicional"]) == 480

    names = list(splits.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = splits[names[i]] & splits[names[j]]
            assert len(overlap) == 0, f"Fuga entre {names[i]} y {names[j]}: {len(overlap)} muestras."

    union_all = set().union(*splits.values())
    assert len(union_all) == 2880


def test_dataset_held_out_pairs_never_appear_in_train():
    cfg = HybridConfig()
    train_ds = ProceduralCompositionalDataset(cfg, split="train")
    for prompt_b, _, _, _ in train_ds.samples:
        lines = dict(ln.split(": ", 1) for ln in prompt_b.decode("utf-8").strip().splitlines() if ": " in ln)
        pair = (lines["perfil"], lines["carga"])
        assert pair not in ProceduralCompositionalDataset.HELD_OUT_PAIRS


def test_dataset_rejects_unknown_split():
    cfg = HybridConfig()
    with pytest.raises(ValueError, match="Split desconocido"):
        ProceduralCompositionalDataset(cfg, split="split-inexistente")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
