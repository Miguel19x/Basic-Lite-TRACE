# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Suite de validación contractual de TRACELanguageModel (MLA ternaria, KV cache, QAT)."""

import pytest
import torch
import torch.nn as nn

from trace.layers.bitlinear import BitLinear
from trace.lm.config import TRACELMConfig
from trace.lm.model import TRACELanguageModel


def test_tied_embeddings_share_parameter_storage():
    cfg = TRACELMConfig(n_layers=2)
    model = TRACELanguageModel(cfg)
    assert model.lm_head.weight is model.tok_emb.weight


def test_all_lm_block_linear_projections_are_bitlinear():
    cfg = TRACELMConfig(n_layers=2)
    model = TRACELanguageModel(cfg)
    bitlinear_count = 0
    for layer in model.layers:
        for name, mod in layer.named_modules():
            assert not isinstance(mod, nn.Linear), f"Proyección nn.Linear FP32 no permitida: {name}"
            if isinstance(mod, BitLinear):
                bitlinear_count += 1
    assert bitlinear_count == cfg.n_layers * 9


def test_weight_absorption_equivalence():
    torch.manual_seed(42)
    cfg = TRACELMConfig(n_layers=2, max_seq_len=64)
    model = TRACELanguageModel(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 16))

    with torch.no_grad():
        logits_std, _, _ = model(x, use_weight_absorption=False)
        logits_abs, _, _ = model(x, use_weight_absorption=True)

    max_diff = (logits_std - logits_abs).abs().max().item()
    assert max_diff < 1e-4, f"Discrepancia estándar vs absorción: {max_diff:.6e}"


def test_prefix_caching_matches_full_prefill():
    torch.manual_seed(123)
    cfg = TRACELMConfig(n_layers=2, max_seq_len=64)
    model = TRACELanguageModel(cfg).eval()
    full_seq = torch.randint(0, cfg.vocab_size, (1, 8))

    with torch.no_grad():
        logits_ref, _, _ = model(full_seq, use_weight_absorption=True)
        _, _, prefix_caches = model(full_seq[:, :4], use_weight_absorption=True)
        logits_suffix, _, _ = model(full_seq[:, 4:], kv_caches=prefix_caches, use_weight_absorption=True)

    diff = (logits_ref[:, 4:] - logits_suffix).abs().max().item()
    assert diff < 1e-4, f"Fallo en prefix caching: {diff:.6e}"


def test_incremental_decoding_matches_full_prefill():
    torch.manual_seed(123)
    cfg = TRACELMConfig(n_layers=2, max_seq_len=64)
    model = TRACELanguageModel(cfg).eval()
    full_seq = torch.randint(0, cfg.vocab_size, (1, 8))

    with torch.no_grad():
        logits_ref, _, _ = model(full_seq, use_weight_absorption=True)
        _, _, curr_caches = model(full_seq[:, :4], use_weight_absorption=True)
        for i in range(4):
            tok_in = full_seq[:, 4 + i : 5 + i]
            logits_step, _, curr_caches = model(tok_in, kv_caches=curr_caches, use_weight_absorption=True)
            diff_step = (logits_ref[:, 4 + i : 5 + i] - logits_step).abs().max().item()
            assert diff_step < 1e-4, f"Fallo en paso incremental {i}: {diff_step:.6e}"


def test_quantize_kv_int8_applies_bounded_quantization():
    torch.manual_seed(42)
    cfg = TRACELMConfig(n_layers=2, max_seq_len=64)
    model = TRACELanguageModel(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 16))

    with torch.no_grad():
        logits_fp32, _, _ = model(x, quantize_kv_int8=False)
        logits_int8, _, _ = model(x, quantize_kv_int8=True)

    assert torch.isfinite(logits_int8).all()
    abs_diff = (logits_fp32 - logits_int8).abs()
    assert abs_diff.max().item() > 0.0, "quantize_kv_int8=True no tuvo efecto (no-op)."
    assert abs_diff.mean().item() < 0.25
    assert abs_diff.max().item() < 1.0


def test_backward_produces_finite_nonzero_gradients():
    torch.manual_seed(42)
    cfg = TRACELMConfig(n_layers=2, d_model=128, n_heads=4, d_ff=256)
    model = TRACELanguageModel(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))

    _, loss, _ = model(x, targets=y)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradiente None en {name}"
            assert torch.isfinite(param.grad).all(), f"Gradiente no finito en {name}"
            assert param.grad.abs().sum().item() > 0.0, f"Gradiente cero en {name}"


def test_five_step_training_reduces_loss():
    torch.manual_seed(42)
    cfg = TRACELMConfig(n_layers=2, d_model=128, n_heads=4, d_ff=256)
    model = TRACELanguageModel(cfg).train()
    optimizer = torch.optim.AdamW(
        model.configure_adamw_groups(bitlinear_weight_decay=0.0, weight_decay=0.01),
        lr=1e-3,
    )
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))

    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        _, loss, _ = model(x, targets=y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    assert all(torch.isfinite(torch.tensor(v)) for v in losses)
    assert losses[-1] < losses[0], f"La pérdida no descendió tras 5 pasos: {losses}"


def test_generate_greedy_equivalence():
    torch.manual_seed(42)
    cfg = TRACELMConfig(n_layers=2)
    model = TRACELanguageModel(cfg).eval()
    prompt = torch.tensor([[cfg.bos_token_id, 100, 200]])
    max_new = 8

    gen_cached = model.generate(prompt, max_new_tokens=max_new, temperature=0.0)

    gen_ref = prompt.clone()
    with torch.no_grad():
        for _ in range(max_new):
            logits, _, _ = model(gen_ref, use_weight_absorption=False)
            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            gen_ref = torch.cat([gen_ref, next_tok], dim=1)

    assert torch.equal(gen_cached, gen_ref)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
