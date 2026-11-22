"""
Suite de Higiene y Contratos E0 para TRACECausalModel:
- E0.1 Causalidad Temporal Diferenciable: d(logits_prompt) / d(target_embed) == 0.0.
  Garantiza que la región de prompt no reciba gradientes ni información del target futuro.
- E0.3 Independencia Estricta de Lote: los logits de una muestra aislada coinciden
  exactamente (diff <= 1e-5) cuando la muestra se procesa dentro de un lote heterogéneo.
"""
import torch
import torch.nn as nn
from config import HybridConfig
from trace.models.causal import TRACECausalModel
from data import ProceduralCompositionalDataset
from train import collate_fn


def test_e0_1_prompt_causality():
    """Verifica que el prompt no atienda a la región de target."""
    torch.manual_seed(42)
    cfg = HybridConfig(device="cpu", num_recursions=1, adaptive_recursion=False)
    model = TRACECausalModel(cfg)
    model.eval()

    dataset = ProceduralCompositionalDataset(cfg, split="dev")
    prompt_b, target_b, _, _ = dataset.samples[0]
    plen = 1 + len(prompt_b)

    seq = [cfg.bos_token_id] + list(prompt_b) + list(target_b) + [cfg.eos_token_id]
    seq = seq + [cfg.pad_token_id] * (cfg.max_seq_len - len(seq))
    x = torch.tensor([seq], dtype=torch.long)
    t = torch.ones(1)

    # Enganchar gradientes en los embeddings continuos
    emb_grad = []
    def hook_fn(module, inp, out):
        out.retain_grad()
        emb_grad.append(out)

    handle = model.token_emb.register_forward_hook(hook_fn)
    model.zero_grad()

    # Reconstruir máscara causal unificada
    from trace.modules.attention import build_target_unified_mask
    attn_mask = build_target_unified_mask([plen], cfg.max_seq_len, cfg.block_size, torch.device("cpu"))

    with torch.enable_grad():
        logits = model.forward_backbone(x, t, stage_id=0, attn_mask=attn_mask)
        # Sumar logits de la región de prompt (0..plen-1)
        loss = logits[:, :plen, :].sum()
        loss.backward()

    handle.remove()
    emb = emb_grad[0]
    # El gradiente sobre los tokens de target (plen..) debe ser exactamente 0.0
    target_grad = emb.grad[:, plen:, :]
    max_leakage = target_grad.abs().max().item()

    assert max_leakage == 0.0, f"Fuga causal detectada: gradiente máximo de {max_leakage} en región de target."


def test_e0_3_batch_independence():
    """Verifica que la inferencia de una muestra no dependa de las otras muestras del lote."""
    torch.manual_seed(42)
    cfg = HybridConfig(device="cpu", num_recursions=1, adaptive_recursion=False)
    model = TRACECausalModel(cfg)
    model.eval()

    dataset = ProceduralCompositionalDataset(cfg, split="dev")
    sample_a = dataset.samples[0]
    sample_b = dataset.samples[1]
    sample_c = dataset.samples[2]
    sample_d = dataset.samples[3]

    # Procesar muestra A en solitario (B=1)
    x_solo, plen_solo = collate_fn([sample_a], cfg.max_seq_len, cfg.pad_token_id, cfg.bos_token_id, cfg.eos_token_id)
    from trace.modules.attention import build_target_unified_mask
    mask_solo = build_target_unified_mask(plen_solo, cfg.max_seq_len, cfg.block_size, torch.device("cpu"))
    with torch.no_grad():
        logits_solo = model.forward_backbone(x_solo, torch.ones(1), stage_id=0, attn_mask=mask_solo)

    # Procesar muestra A dentro de un lote de 4 muestras heterogéneas
    batch_samples = [sample_a, sample_b, sample_c, sample_d]
    x_batch, plen_batch = collate_fn(batch_samples, cfg.max_seq_len, cfg.pad_token_id, cfg.bos_token_id, cfg.eos_token_id)
    mask_batch = build_target_unified_mask(plen_batch, cfg.max_seq_len, cfg.block_size, torch.device("cpu"))
    with torch.no_grad():
        logits_batch = model.forward_backbone(x_batch, torch.ones(4), stage_id=0, attn_mask=mask_batch)

    # Comparar logits de la muestra 0 (muestra A)
    # Ambas pueden haber sido recortadas a L_trim; verificar hasta la longitud común
    L_common = min(logits_solo.shape[1], logits_batch.shape[1])
    diff = (logits_solo[0, :L_common] - logits_batch[0, :L_common]).abs().max().item()

    assert diff <= 1e-5, f"Violación de independencia de lote: discrepancia máxima de {diff} > 1e-5."


if __name__ == "__main__":
    test_e0_1_prompt_causality()
    print("✓ E0.1 Causalidad temporal (sin fugas del prompt al target) superada.")
    test_e0_3_batch_independence()
    print("✓ E0.3 Independencia estricta de lote superada.")
    print("✓ Todas las pruebas de higiene E0 pasaron exitosamente.")
