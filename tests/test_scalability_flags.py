"""
Test de regresión para las 4 banderas del Sistema de Activación por Escalabilidad (TRACE Núcleo 870K -> 30B).
Verifica:
1. Valores por defecto (todas las banderas inactivas a escala 870K).
2. Sincronización automática de HybridConfig.__post_init__ (archive_context_enabled = enable_sqlite_context_l3).
3. Conmutación correcta de módulos:
   - use_mla=False -> BlockCausalGQATernaryAttention
   - use_mla=True -> MultiHeadLatentAttention
   - enable_nvme_io_simulation=True/False en PagedMoE
"""
from config import HybridConfig
from trace.models.causal import TRACECausalModel
from trace.modules.attention import BlockCausalGQATernaryAttention
from trace.modules.mla_attention import MultiHeadLatentAttention


def test_default_scalability_flags_off():
    cfg = HybridConfig()
    assert cfg.enable_nvme_io_simulation is False, "enable_nvme_io_simulation debe ser False por defecto en 870K"
    assert cfg.enable_sqlite_context_l3 is False, "enable_sqlite_context_l3 debe ser False por defecto en 870K"
    assert cfg.use_mla is False, "use_mla debe ser False por defecto en 870K"
    assert cfg.use_byte_patching is False, "use_byte_patching debe ser False por defecto en 870K"
    assert cfg.archive_context_enabled is False, "archive_context_enabled debe sincronizarse a False por defecto"


def test_post_init_sqlite_sync():
    cfg_active = HybridConfig(enable_sqlite_context_l3=True)
    assert cfg_active.archive_context_enabled is True, "archive_context_enabled debe sincronizarse con enable_sqlite_context_l3=True"

    cfg_inactive = HybridConfig(enable_sqlite_context_l3=False)
    assert cfg_inactive.archive_context_enabled is False, "archive_context_enabled debe sincronizarse con enable_sqlite_context_l3=False"


def test_attention_module_switching():
    # 870K: GQA directa estándar
    cfg_gqa = HybridConfig(use_mla=False, num_recursions=1, adaptive_recursion=False)
    model_gqa = TRACECausalModel(cfg_gqa)
    assert isinstance(model_gqa.resident_core.attn, BlockCausalGQATernaryAttention), (
        "Con use_mla=False, la atención debe ser BlockCausalGQATernaryAttention"
    )

    # 30B: Compresión KV Latente (MLA)
    cfg_mla = HybridConfig(use_mla=True, num_recursions=1, adaptive_recursion=False)
    model_mla = TRACECausalModel(cfg_mla)
    assert isinstance(model_mla.resident_core.attn, MultiHeadLatentAttention), (
        "Con use_mla=True, la atención debe ser MultiHeadLatentAttention"
    )


def test_nvme_simulation_flag():
    cfg_off = HybridConfig(enable_nvme_io_simulation=False, num_recursions=1, adaptive_recursion=False)
    model_off = TRACECausalModel(cfg_off)
    assert model_off.resident_core.paged_moe.enable_nvme_io_simulation is False

    cfg_on = HybridConfig(enable_nvme_io_simulation=True, num_recursions=1, adaptive_recursion=False)
    model_on = TRACECausalModel(cfg_on)
    assert model_on.resident_core.paged_moe.enable_nvme_io_simulation is True


if __name__ == "__main__":
    test_default_scalability_flags_off()
    test_post_init_sqlite_sync()
    test_attention_module_switching()
    test_nvme_simulation_flag()
    print("✓ Todos los tests de regresión de banderas de escalabilidad pasaron exitosamente.")
