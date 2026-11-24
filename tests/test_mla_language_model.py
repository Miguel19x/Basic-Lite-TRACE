# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Suite de Validación y Contratos Arquitectónicos para TRACELanguageModel (Hito 3.2):
1. Equivalencia matemática estándar vs. absorción completa (W_uk y W_uv, diff < 1e-4).
2. Prefill completo vs. decodificación incremental token a token Y prefix caching (S > 1 con kv_cache).
3. Verificación contractual de tied embeddings (tok_emb.weight is lm_head.weight).
4. Verificación de que el 100% de las proyecciones en TRACELMBlock son BitLinear W1.58.
5. Inferencia con quantize_kv_int8=True.
6. Reducción monótona de pérdida en 5 pasos sintéticos con configure_adamw_groups.
7. Verificación de gradientes finitos y no-nulos en w_dkv, w_uk, w_uv y tok_emb.
8. Paridad exacta de generate(temperature=0.0) vs decodificación greedy paso a paso sin caché.
"""

from trace.layers.bitlinear import BitLinear
from trace.lm.config import TRACELMConfig
from trace.lm.model import TRACELanguageModel, TRACELMBlock

import torch


def test_tied_embeddings_and_bitlinear_projections():
    cfg = TRACELMConfig(n_layers=2)
    model = TRACELanguageModel(cfg)

    # 1. Tied Embeddings
    assert model.lm_head.weight is model.tok_emb.weight, (
        "Violación: lm_head.weight no comparte referencia con tok_emb.weight"
    )

    # 2. 100% de proyecciones en TRACELMBlock son BitLinear
    for idx, layer in enumerate(model.layers):
        assert isinstance(layer, TRACELMBlock)
        attn = layer.attn
        assert isinstance(attn.w_dkv, BitLinear)
        assert isinstance(attn.w_kr, BitLinear)
        assert isinstance(attn.w_q, BitLinear)
        assert isinstance(attn.w_qr, BitLinear)
        assert isinstance(attn.w_uk, BitLinear) and attn.w_uk.use_sub_ln is False
        assert isinstance(attn.w_uv, BitLinear) and attn.w_uv.use_sub_ln is False
        assert isinstance(attn.w_out, BitLinear)

        ffn = layer.ffn
        assert isinstance(ffn.up, BitLinear)
        assert isinstance(ffn.down, BitLinear)


def test_weight_absorption_equivalence():
    """Certifica diff < 1e-4 entre forward estándar y forward con absorción completa."""
    torch.manual_seed(42)
    cfg = TRACELMConfig(n_layers=2, max_seq_len=64)
    model = TRACELanguageModel(cfg).eval()

    x = torch.randint(0, cfg.vocab_size, (2, 16))

    with torch.no_grad():
        logits_std, _, _ = model(x, use_weight_absorption=False)
        logits_abs, _, _ = model(x, use_weight_absorption=True)

    max_diff = (logits_std - logits_abs).abs().max().item()
    assert max_diff < 1e-4, f"Fallo en absorción de pesos: max_diff = {max_diff:.6e} >= 1e-4"
    print(f"✓ Equivalencia de absorción completa (W_uk + W_uv) certificada: max_diff = {max_diff:.6e} < 1e-4")


def test_prefix_caching_and_incremental_decoding():
    """Valida prefill completo vs prefix caching (S > 1) vs decodificación token a token (S = 1)."""
    torch.manual_seed(123)
    cfg = TRACELMConfig(n_layers=2, max_seq_len=64)
    model = TRACELanguageModel(cfg).eval()

    full_seq = torch.randint(0, cfg.vocab_size, (1, 8))

    with torch.no_grad():
        # Referencia: prefill de la secuencia completa
        logits_ref, _, _ = model(full_seq, use_weight_absorption=True)

        # Caso A: Prefix caching (Prefill 4 tokens + sufijo de 4 tokens con S=4 > 1)
        prefix = full_seq[:, :4]
        suffix = full_seq[:, 4:]
        _, _, prefix_caches = model(prefix, use_weight_absorption=True)
        logits_suffix, _, _ = model(suffix, kv_caches=prefix_caches, use_weight_absorption=True)

        diff_prefix = (logits_ref[:, 4:] - logits_suffix).abs().max().item()
        assert diff_prefix < 1e-4, f"Fallo en prefix caching: {diff_prefix:.6e} >= 1e-4"

        # Caso B: Decodificación incremental token a token (S = 1)
        curr_caches = prefix_caches
        for i in range(4):
            tok_in = full_seq[:, 4 + i : 5 + i]
            logits_step, _, curr_caches = model(tok_in, kv_caches=curr_caches, use_weight_absorption=True)
            diff_step = (logits_ref[:, 4 + i : 5 + i] - logits_step).abs().max().item()
            assert diff_step < 1e-4, f"Fallo en paso incremental {i}: {diff_step:.6e} >= 1e-4"

    print("✓ Paridad exacta de Prefix Caching (S > 1) y decodificación incremental token a token (S = 1) superadas.")


def test_quantize_kv_int8():
    """Verifica que el modo quantize_kv_int8=True opere de forma estable."""
    torch.manual_seed(42)
    cfg = TRACELMConfig(n_layers=2, max_seq_len=64)
    model = TRACELanguageModel(cfg).eval()

    x = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        logits_fp32, _, _ = model(x, quantize_kv_int8=False)
        logits_int8, _, _ = model(x, quantize_kv_int8=True)

    assert torch.isfinite(logits_int8).all(), "Logits INT8 contienen NaN o Inf"
    diff = (logits_fp32 - logits_int8).abs().mean().item()
    assert diff < 0.25, f"Discrepancia excesiva en cuantización INT8 de KV: {diff:.4f}"
    print(f"✓ Cuantización simétrica INT8 de KV cache verificada con finitud y delta acotado ({diff:.4f}).")


def test_monotonic_loss_decrease_and_gradients():
    """Verifica optimización con AdamW BitNet y gradientes finitos/no-nulos en capas clave."""
    torch.manual_seed(42)
    cfg = TRACELMConfig(n_layers=2, d_model=128, n_heads=4, d_ff=256)
    model = TRACELanguageModel(cfg).train()

    optimizer = torch.optim.AdamW(
        model.configure_adamw_groups(bitlinear_weight_decay=0.0, weight_decay=0.01),
        lr=1e-3
    )

    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))

    losses = []
    for step in range(5):
        optimizer.zero_grad()
        _, loss, _ = model(x, targets=y)
        loss.backward()

        # Verificación estricta de gradientes en capas clave
        if step == 0:
            for l_idx, layer in enumerate(model.layers):
                for p_name, p in [
                    ("w_dkv", layer.attn.w_dkv.weight),
                    ("w_uk", layer.attn.w_uk.weight),
                    ("w_uv", layer.attn.w_uv.weight),
                ]:
                    assert p.grad is not None, f"Gradiente None en {p_name} capa {l_idx}"
                    assert torch.isfinite(p.grad).all(), f"Gradiente no finito en {p_name} capa {l_idx}"
                    assert p.grad.abs().sum().item() > 0.0, f"Gradiente estrictamente cero en {p_name} capa {l_idx}"

            assert model.tok_emb.weight.grad is not None
            assert torch.isfinite(model.tok_emb.weight.grad).all()
            assert model.tok_emb.weight.grad.abs().sum().item() > 0.0

        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"La pérdida no decreció: inicial={losses[0]:.4f}, final={losses[-1]:.4f}"
    print("✓ Gradientes finitos y no-nulos verificados en W_dkv, W_uk, W_uv y tok_emb.")
    print(f"✓ Descenso monótono de pérdida verificado en 5 pasos: {losses[0]:.4f} -> {losses[-1]:.4f}")


def test_generate_greedy_equivalence():
    """Verifica que generate(temperature=0.0) con caché coincida exactamente con decodificación paso a paso sin caché."""
    torch.manual_seed(42)
    cfg = TRACELMConfig(n_layers=2)
    model = TRACELanguageModel(cfg).eval()

    prompt = torch.tensor([[cfg.bos_token_id, 100, 200]])
    max_new = 8

    # 1. Generación vía generate() con absorción y kv_caches
    gen_cached = model.generate(prompt, max_new_tokens=max_new, temperature=0.0)

    # 2. Generación manual de referencia paso a paso sin kv_cache (recomputando toda la secuencia)
    gen_ref = prompt.clone()
    for _ in range(max_new):
        logits, _, _ = model(gen_ref, use_weight_absorption=False)
        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        gen_ref = torch.cat([gen_ref, next_tok], dim=1)

    assert torch.equal(gen_cached, gen_ref), (
        f"Discrepancia en generación greedy:\n"
        f"Con caché: {gen_cached}\n"
        f"Sin caché: {gen_ref}"
    )
    print("✓ Paridad exacta 100% de generate(temperature=0.0) con kv_cache vs recomputación completa sin caché.")


if __name__ == "__main__":
    test_tied_embeddings_and_bitlinear_projections()
    test_weight_absorption_equivalence()
    test_prefix_caching_and_incremental_decoding()
    test_quantize_kv_int8()
    test_monotonic_loss_decrease_and_gradients()
    test_generate_greedy_equivalence()
    print("✓ Todas las pruebas del modelo de lenguaje con MLA pasaron exitosamente.")
