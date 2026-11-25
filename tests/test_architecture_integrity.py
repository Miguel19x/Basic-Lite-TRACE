# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Suite de Integridad Arquitectónica — Basic-Lite TRACE:
1. Contrato de configuración por defecto de producción (2 hilos, 4096 B, fast).
2. Sincronización contractual SQLite FTS5 L3 en HybridConfig.__post_init__.
3. Paridad numérica FP32 vs add_sub_mode en TernaryExpert.
4. Comportamiento de MemoryAwarePagedMoE (paridad add_sub, máscara de padding, telemetría).
5. Determinismo en eval() de TRACECausalModel.forward_backbone.
"""

import pytest
import torch

from config import HybridConfig
from trace.layers.ffn import TernaryExpert
from trace.models.causal import TRACECausalModel
from trace.modules.paged_moe import MemoryAwarePagedMoE


def test_default_production_config():
    cfg = HybridConfig()
    assert cfg.cpu_threads == 2
    assert cfg.max_seq_len == 160
    assert cfg.d_model == 192
    assert cfg.paged_bitcos_page_size == 4096
    assert cfg.paged_bitcos_format == "fast"
    assert cfg.archive_context_enabled is False


@pytest.mark.parametrize("enabled", [True, False])
def test_sqlite_fts5_flag_sync(enabled: bool):
    cfg = HybridConfig(enable_sqlite_context_l3=enabled)
    assert cfg.archive_context_enabled is enabled


def test_ternary_expert_addsub_matches_fp32():
    torch.manual_seed(42)
    expert = TernaryExpert(d_model=192, d_ff=384).eval()
    x = torch.randn(2, 16, 192)
    with torch.no_grad():
        y_fp32 = expert(x, add_sub_mode=False)
        y_addsub = expert(x, add_sub_mode=True)

    assert y_fp32.shape == (2, 16, 192)
    assert y_addsub.shape == (2, 16, 192)
    diff = float((y_fp32 - y_addsub).abs().max().item())
    assert diff < 1e-4, f"Discrepancia FP32 vs add_sub_mode=True en TernaryExpert: {diff:.6e}"


def test_paged_moe_addsub_matches_fp32():
    torch.manual_seed(42)
    cfg = HybridConfig(num_recursions=1)
    moe = MemoryAwarePagedMoE(cfg).eval()
    x = torch.randn(2, 16, 192)
    t = torch.ones(2, 16)

    with torch.no_grad():
        out_fp32, nodes_fp32 = moe(x, t, add_sub_mode=False)
        out_int, nodes_int = moe(x, t, integer_mode=True)
        out_addsub, nodes_addsub = moe(x, t, add_sub_mode=True)

    assert out_fp32.shape == (2, 16, 192)
    assert torch.equal(nodes_fp32, nodes_addsub)
    assert torch.equal(nodes_int, nodes_addsub)
    diff_int = float((out_int - out_addsub).abs().max().item())
    diff_fp32 = float((out_fp32 - out_addsub).abs().max().item())
    assert diff_int < 1e-5, f"Discrepancia integer_mode vs add_sub en MoE: {diff_int:.6e}"
    assert diff_fp32 < 5e-3, f"Discrepancia FP32 vs add_sub en MoE: {diff_fp32:.6e}"

    telemetry = moe.get_last_telemetry()
    assert len(telemetry["selected_experts"]) >= 1
    assert 0.0 < telemetry["active_slot_ratio"] <= 1.0


def test_paged_moe_zeros_masked_padding_tokens():
    torch.manual_seed(42)
    cfg = HybridConfig(num_recursions=1)
    moe = MemoryAwarePagedMoE(cfg).eval()
    x = torch.randn(1, 16, 192)
    t = torch.ones(1, 16)
    non_pad_mask = torch.zeros(1, 16, dtype=torch.bool)
    non_pad_mask[:, :8] = True

    with torch.no_grad():
        out, _ = moe(x, t, non_pad_mask=non_pad_mask)

    assert out[:, 8:, :].abs().max().item() == 0.0, "Posiciones de padding deben quedar en cero."
    assert out[:, :8, :].abs().max().item() > 0.0, "Posiciones activas no deben colapsar a cero."


def test_trace_causal_model_eval_is_deterministic():
    torch.manual_seed(42)
    cfg = HybridConfig(device="cpu", num_recursions=1, adaptive_recursion=False)
    model = TRACECausalModel(cfg).eval()

    x = torch.randint(0, cfg.vocab_size, (1, 32), dtype=torch.long)
    t = torch.ones(1)
    with torch.no_grad():
        logits_a = model.forward_backbone(x, t, stage_id=0)
        logits_b = model.forward_backbone(x, t, stage_id=0)

    assert logits_a.shape == (1, 32, cfg.vocab_size)
    assert torch.isfinite(logits_a).all()
    assert torch.equal(logits_a, logits_b), "forward_backbone en eval() debe ser determinista."


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
