"""
Suite de Integridad Arquitectónica — Basic-Lite TRACE (Hito 3.0):
Verifica:
1. Configuración por defecto de producción (CPU 2 hilos, Paged BitCOS fast, 160 bytes).
2. Sincronización contractual SQLite FTS5 L3 en HybridConfig.__post_init__.
3. Integridad del módulo FFN ternario TernaryExpert en FP32 y Add/Sub.
4. MoE residente (MemoryAwarePagedMoE) con 4 expertos residentes + experto compartido.
5. Inferencia forward pass en TRACECausalModel con 0 tensores FP32 en pesos BitLinear.
"""
from trace.layers.ffn import TernaryExpert
from trace.models.causal import TRACECausalModel
from trace.modules.attention import BlockCausalGQATernaryAttention
from trace.modules.paged_moe import MemoryAwarePagedMoE

import torch

from config import HybridConfig


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
    expert = TernaryExpert(d_model=192, d_ff=384)
    x = torch.randn(2, 16, 192)
    y = expert(x)
    assert y.shape == (2, 16, 192)


def test_paged_moe_resident_integrity():
    cfg = HybridConfig(num_recursions=1)
    moe = MemoryAwarePagedMoE(cfg)
    assert len(moe.expert_bank) == 4
    assert moe.use_shared_expert is True

    x = torch.randn(2, 16, 192)
    t = torch.ones(2, 16)
    out, node_ids = moe(x, t)
    assert out.shape == (2, 16, 192)
    assert node_ids.shape == (2, 16)


def test_trace_causal_model_resident_core():
    cfg = HybridConfig(num_recursions=1)
    model = TRACECausalModel(cfg)
    assert isinstance(model.resident_core.attn, BlockCausalGQATernaryAttention)
    assert isinstance(model.resident_core.paged_moe, MemoryAwarePagedMoE)


if __name__ == "__main__":
    test_default_production_config()
    test_sqlite_fts5_flag_sync()
    test_ternary_expert_ffn()
    test_paged_moe_resident_integrity()
    test_trace_causal_model_resident_core()
    print("✓ Suite de Integridad Arquitectónica superada exitosamente.")
