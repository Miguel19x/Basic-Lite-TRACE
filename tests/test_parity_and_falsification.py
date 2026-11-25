# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Suite de paridad numérica y falsación activa para Paged BitCOS v1 (TRACECausalModel)."""

import copy
import struct

import pytest
import torch

from config import HybridConfig
from data import ProceduralCompositionalDataset
from tests._helpers import (
    PAGED_HEADER_FMT,
    PAGED_HEADER_SIZE,
    REAL_CAUSAL_BIN,
    REAL_CAUSAL_CKPT,
    flip_ternary_payload_inplace,
    has_real_causal_checkpoints,
)
from trace.models.causal import TRACECausalModel
from trace.serialization.paged_bitcos import (
    PagedBitCOSReader,
    PagedBitCOSSerializer,
    count_fp32_weights_in_bitlinear,
)


def _build_synthetic_causal_fixture(tmp_path):
    torch.manual_seed(42)
    cfg = HybridConfig(device="cpu", cpu_threads=2, num_recursions=1, adaptive_recursion=False)
    cfg.use_add_sub_inference = True
    base = TRACECausalModel(cfg).eval()
    bin_path = str(tmp_path / "synthetic_causal.bin")
    PagedBitCOSSerializer.export_paged_model(base, bin_path, format_type="fast")

    dataset = ProceduralCompositionalDataset(cfg, split="dev")
    prompt_b = dataset.samples[0][0]
    template_target = b"pw: ???\npms: ???\nwatt: ???\nms: ???\n"
    seq = [cfg.bos_token_id] + list(prompt_b) + list(template_target) + [cfg.eos_token_id]
    seq = seq + [cfg.pad_token_id] * max(0, cfg.max_seq_len - len(seq))
    x = torch.tensor([seq], dtype=torch.long)
    return cfg, base, bin_path, prompt_b, x


def _load_paged_model(cfg: HybridConfig, bin_path: str) -> TRACECausalModel:
    model = TRACECausalModel(cfg)
    reader = PagedBitCOSReader(bin_path)
    try:
        reader.load_into_model(model)
    finally:
        reader.close()
    return model.eval()


def test_synthetic_paged_bitcos_matches_pytorch_addsub(tmp_path):
    cfg, model_ref, bin_path, prompt_b, x = _build_synthetic_causal_fixture(tmp_path)
    model_paged = _load_paged_model(cfg, bin_path)

    assert count_fp32_weights_in_bitlinear(model_paged) == 0

    with torch.no_grad():
        out_ref, _ = model_ref.generate_causal_two_step(prompt_b)
        out_paged, _ = model_paged.generate_causal_two_step(prompt_b)
    assert out_paged == out_ref

    t = torch.ones(1)
    for stage_id in (0, 1, 2, 3):
        with torch.no_grad():
            logits_ref = model_ref.forward_backbone(x, t, stage_id=stage_id)
            logits_paged = model_paged.forward_backbone(x, t, stage_id=stage_id)
        diff = float((logits_ref - logits_paged).abs().max().item())
        assert diff <= 1e-4, f"Discrepancia etapa {stage_id}: {diff:.6e}"


def test_falsification_sign_flip_degrades_logits(tmp_path):
    cfg, _, bin_path, _, x = _build_synthetic_causal_fixture(tmp_path)
    model_sano = _load_paged_model(cfg, bin_path)
    t = torch.ones(1)
    with torch.no_grad():
        logits_sano = model_sano.forward_backbone(x, t, stage_id=0)

    reader = PagedBitCOSReader(bin_path)
    try:
        page_sz, pgs_off = reader.page_size, reader.pages_offset
    finally:
        reader.close()

    with open(bin_path, "rb") as f:
        mutated = bytearray(f.read())
    flip_ternary_payload_inplace(mutated, pgs_off + PAGED_HEADER_SIZE, pgs_off + page_sz)

    mut_path = tmp_path / "falsified_sign_flip.bin"
    mut_path.write_bytes(mutated)
    model_mut = _load_paged_model(cfg, str(mut_path))

    with torch.no_grad():
        logits_mut = model_mut.forward_backbone(x, t, stage_id=0)
    assert float((logits_sano - logits_mut).abs().max().item()) > 0.1


def test_falsification_scale_alteration_degrades_logits(tmp_path):
    cfg, _, bin_path, _, x = _build_synthetic_causal_fixture(tmp_path)
    model_sano = _load_paged_model(cfg, bin_path)
    t = torch.ones(1)
    with torch.no_grad():
        logits_sano = model_sano.forward_backbone(x, t, stage_id=0)

    reader = PagedBitCOSReader(bin_path)
    try:
        pgs_off = reader.pages_offset
    finally:
        reader.close()

    with open(bin_path, "rb") as f:
        mutated = bytearray(f.read())
    scale_offset = pgs_off + 12
    orig_scale = struct.unpack_from("<f", mutated, scale_offset)[0]
    struct.pack_into("<f", mutated, scale_offset, orig_scale * 10.0)

    mut_path = tmp_path / "falsified_scale.bin"
    mut_path.write_bytes(mutated)
    model_mut = _load_paged_model(cfg, str(mut_path))

    with torch.no_grad():
        logits_mut = model_mut.forward_backbone(x, t, stage_id=0)
    assert float((logits_sano - logits_mut).abs().max().item()) > 0.1


def test_falsification_page_permutation_degrades_logits(tmp_path):
    cfg, _, bin_path, _, x = _build_synthetic_causal_fixture(tmp_path)
    model_sano = _load_paged_model(cfg, bin_path)
    t = torch.ones(1)
    with torch.no_grad():
        logits_sano = model_sano.forward_backbone(x, t, stage_id=0)

    reader = PagedBitCOSReader(bin_path)
    try:
        page_sz, pgs_off, total_pages = reader.page_size, reader.pages_offset, reader.total_pages
    finally:
        reader.close()

    with open(bin_path, "rb") as f:
        mutated = bytearray(f.read())

    _, rows0, cols0, fmt0, _, _, _ = struct.unpack_from(PAGED_HEADER_FMT, mutated, pgs_off)
    partner_off = None
    for p_idx in range(1, total_pages):
        cand_off = pgs_off + p_idx * page_sz
        _, rows_c, cols_c, fmt_c, _, _, _ = struct.unpack_from(PAGED_HEADER_FMT, mutated, cand_off)
        if (rows_c, cols_c, fmt_c) == (rows0, cols0, fmt0):
            partner_off = cand_off
            break
    assert partner_off is not None

    p0_payload = bytes(mutated[pgs_off + PAGED_HEADER_SIZE : pgs_off + page_sz])
    p1_payload = bytes(mutated[partner_off + PAGED_HEADER_SIZE : partner_off + page_sz])
    mutated[pgs_off + PAGED_HEADER_SIZE : pgs_off + page_sz] = p1_payload
    mutated[partner_off + PAGED_HEADER_SIZE : partner_off + page_sz] = p0_payload

    mut_path = tmp_path / "falsified_permute.bin"
    mut_path.write_bytes(mutated)
    model_mut = _load_paged_model(cfg, str(mut_path))

    with torch.no_grad():
        logits_mut = model_mut.forward_backbone(x, t, stage_id=0)
    assert float((logits_sano - logits_mut).abs().max().item()) > 0.1


@pytest.mark.skipif(not has_real_causal_checkpoints(), reason="Checkpoints locales Fase 1 ausentes.")
def test_real_checkpoint_parity_matches_fp32_text():
    ckpt = torch.load(REAL_CAUSAL_CKPT, map_location="cpu", weights_only=False)
    cfg_fp32 = copy.deepcopy(ckpt["cfg"])
    cfg_fp32.device = "cpu"
    cfg_fp32.cpu_threads = 2
    cfg_fp32.use_add_sub_inference = False

    model_fp32 = TRACECausalModel(cfg_fp32)
    model_fp32.load_state_dict(ckpt["model_state"], strict=True)
    model_fp32.eval()

    cfg_paged = copy.deepcopy(cfg_fp32)
    cfg_paged.use_add_sub_inference = True
    model_paged = _load_paged_model(cfg_paged, REAL_CAUSAL_BIN)
    assert count_fp32_weights_in_bitlinear(model_paged) == 0

    for split_name in ("dev", "test-composicional"):
        dataset = ProceduralCompositionalDataset(cfg_fp32, split=split_name)
        for i in range(4):
            prompt_b = dataset.samples[i][0]
            with torch.no_grad():
                out_fp32, _ = model_fp32.generate_causal_two_step(prompt_b)
                out_paged, _ = model_paged.generate_causal_two_step(prompt_b)
            assert out_paged == out_fp32, f"Discrepancia textual en {split_name}[{i}]"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
