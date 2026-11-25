# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Suite de Higiene y Contratos E0 para TRACECausalModel:
- E0.1 Causalidad Temporal Diferenciable: d(logits_prompt) / d(target_embed) == 0.0.
- E0.3 Independencia Estricta de Lote en todas las etapas (0..3): diff <= 1e-5.
"""

import pytest
import torch

from config import HybridConfig
from data import ProceduralCompositionalDataset
from trace.models.causal import TRACECausalModel
from trace.modules.attention import build_target_unified_mask
from train import collate_fn


@pytest.mark.parametrize("stage_id", [0, 1, 2, 3])
def test_e0_1_prompt_causality(stage_id: int):
    """Verifica en cada etapa que el prompt no atienda al target y sí reciba gradiente no-nulo."""
    torch.manual_seed(42)
    cfg = HybridConfig(device="cpu", num_recursions=1, adaptive_recursion=False)
    model = TRACECausalModel(cfg).eval()

    dataset = ProceduralCompositionalDataset(cfg, split="dev")
    prompt_b, target_b, _, _ = dataset.samples[0]
    plen = 1 + len(prompt_b)

    seq = [cfg.bos_token_id] + list(prompt_b) + list(target_b) + [cfg.eos_token_id]
    seq = seq + [cfg.pad_token_id] * (cfg.max_seq_len - len(seq))
    x = torch.tensor([seq], dtype=torch.long)
    t = torch.ones(1)
    attn_mask = build_target_unified_mask([plen], cfg.max_seq_len, cfg.block_size, torch.device("cpu"))

    emb_grad = []

    def hook_fn(_module, _inp, out):
        out.retain_grad()
        emb_grad.append(out)

    handle = model.token_emb.register_forward_hook(hook_fn)
    try:
        model.zero_grad()
        with torch.enable_grad():
            logits = model.forward_backbone(x, t, stage_id=stage_id, attn_mask=attn_mask)
            loss = logits[:, :plen, :].sum()
            loss.backward()
    finally:
        handle.remove()

    emb = emb_grad[0]
    max_prompt_grad = emb.grad[:, :plen, :].abs().max().item()
    max_leakage = emb.grad[:, plen:, :].abs().max().item()

    assert max_prompt_grad > 0.0, f"Falso positivo en etapa {stage_id}: gradiente en prompt == 0.0."
    assert max_leakage == 0.0, f"Fuga causal en etapa {stage_id}: max_leakage={max_leakage}."


@pytest.mark.parametrize("stage_id", [0, 1, 2, 3])
def test_e0_3_batch_independence(stage_id: int):
    """Verifica que los logits de una muestra aislada coincidan dentro de un lote heterogéneo."""
    torch.manual_seed(42)
    cfg = HybridConfig(device="cpu", num_recursions=1, adaptive_recursion=False)
    model = TRACECausalModel(cfg).eval()

    dataset = ProceduralCompositionalDataset(cfg, split="dev")
    batch_samples = [dataset.samples[i] for i in range(4)]

    x_solo, plen_solo = collate_fn(
        [batch_samples[0]], cfg.max_seq_len, cfg.pad_token_id, cfg.bos_token_id, cfg.eos_token_id
    )
    mask_solo = build_target_unified_mask(plen_solo, cfg.max_seq_len, cfg.block_size, torch.device("cpu"))

    x_batch, plen_batch = collate_fn(
        batch_samples, cfg.max_seq_len, cfg.pad_token_id, cfg.bos_token_id, cfg.eos_token_id
    )
    mask_batch = build_target_unified_mask(plen_batch, cfg.max_seq_len, cfg.block_size, torch.device("cpu"))

    with torch.no_grad():
        logits_solo = model.forward_backbone(x_solo, torch.ones(1), stage_id=stage_id, attn_mask=mask_solo)
        logits_batch = model.forward_backbone(x_batch, torch.ones(4), stage_id=stage_id, attn_mask=mask_batch)

    common_len = min(logits_solo.shape[1], logits_batch.shape[1])
    diff = (logits_solo[0, :common_len] - logits_batch[0, :common_len]).abs().max().item()
    assert diff <= 1e-5, f"Violación de independencia de lote en etapa {stage_id}: diff={diff:.6e} > 1e-5."


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
