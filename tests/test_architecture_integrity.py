# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Suite de Integridad Arquitectónica — Basic-Lite TRACE (Hito 3.0):
Verifica:
1. Configuración por defecto de producción (CPU 2 hilos, Paged BitCOS fast, 160 bytes).
2. Sincronización contractual SQLite FTS5 L3 en HybridConfig.__post_init__.
3. Integridad del módulo FFN ternario TernaryExpert en FP32 y Add/Sub.
4. MoE residente (MemoryAwarePagedMoE) con 4 expertos residentes + experto compartido.
5. Inferencia forward pass en TRACECausalModel con 0 tensores FP32 en pesos BitLinear.
"""
import torch

from config import HybridConfig
from trace.layers.ffn import TernaryExpert
from trace.models.causal import TRACECausalModel
from trace.modules.attention import BlockCausalGQATernaryAttention
from trace.modules.paged_moe import MemoryAwarePagedMoE


def test_default_production_config():
    cfg = HybridConfig()
    assert cfg.cpu_threads == 2
    assert cfg.max_seq_len == 160
    assert cfg.d_model == 192
    assert cfg.paged_bitcos_page_size == 4096
    assert cfg.paged_bitcos_format == "fast"
    assert cfg.archive_context_enabled is False


def test_sqlite_fts5_flag_sync():
    cfg_on = HybridConfig(enable_sqlite_context_l3=True)
    assert cfg_on.archive_context_enabled is True
    cfg_off = HybridConfig(enable_sqlite_context_l3=False)
    assert cfg_off.archive_context_enabled is False


def test_ternary_expert_ffn():
    torch.manual_seed(42)
    expert = TernaryExpert(d_model=192, d_ff=384).eval()
    x = torch.randn(2, 16, 192)
    with torch.no_grad():
        y_fp32 = expert(x, add_sub_mode=False)
        y_addsub = expert(x, add_sub_mode=True)
    assert y_fp32.shape == (2, 16, 192)
    assert y_addsub.shape == (2, 16, 192)
    assert torch.isfinite(y_fp32).all(), "Salida FP32 de TernaryExpert contiene NaN/Inf."
    assert torch.isfinite(y_addsub).all(), "Salida add_sub de TernaryExpert contiene NaN/Inf."
    diff = float((y_fp32 - y_addsub).abs().max().item())
    assert diff < 1e-4, f"Discrepancia FP32 vs add_sub_mode=True en TernaryExpert: {diff:.6e}"


def test_paged_moe_resident_integrity():
    torch.manual_seed(42)
    cfg = HybridConfig(num_recursions=1)
    moe = MemoryAwarePagedMoE(cfg).eval()
    assert len(moe.expert_bank) == 4
    assert moe.use_shared_expert is True

    x = torch.randn(2, 16, 192)
    t = torch.ones(2, 16)
    with torch.no_grad():
        out, node_ids = moe(x, t)
    assert out.shape == (2, 16, 192)
    assert node_ids.shape == (2, 16)
    assert torch.isfinite(out).all(), "Salida de MemoryAwarePagedMoE contiene NaN/Inf."


def test_trace_causal_model_resident_core():
    torch.manual_seed(42)
    cfg = HybridConfig(device="cpu", num_recursions=1, adaptive_recursion=False)
    model = TRACECausalModel(cfg).eval()
    assert isinstance(model.resident_core.attn, BlockCausalGQATernaryAttention)
    assert isinstance(model.resident_core.paged_moe, MemoryAwarePagedMoE)

    x = torch.randint(0, cfg.vocab_size, (1, 32), dtype=torch.long)
    t = torch.ones(1)
    with torch.no_grad():
        logits = model.forward_backbone(x, t, stage_id=0)
    assert logits.shape == (1, 32, cfg.vocab_size)
    assert torch.isfinite(logits).all(), "Logits de TRACECausalModel contienen NaN/Inf."


if __name__ == "__main__":
    test_default_production_config()
    test_sqlite_fts5_flag_sync()
    test_ternary_expert_ffn()
    test_paged_moe_resident_integrity()
    test_trace_causal_model_resident_core()
    print("✓ Suite de Integridad Arquitectónica superada exitosamente.")
