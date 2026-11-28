# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Motor de entrenamiento Fase 5: SFT enmascarado sobre completado (<think> + TIR)
y GRPO / RLVR corto con regularización KL no negativa (estimador de Schulman).
"""

from __future__ import annotations

import types
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from trace.reasoning.rewards import compute_grpo_advantages, score_reasoning_completion


def quantize_w158_ste(w: torch.Tensor) -> torch.Tensor:
    """Cuantización ternaria BitNet b1.58 con Straight-Through Estimator (idéntica a import_bitnet.py)."""
    scale = w.detach().float().abs().mean().clamp_min(1e-5).to(w.dtype)
    w_q = torch.clamp(torch.round(w / scale), -1.0, 1.0) * scale
    return w + (w_q - w).detach()


def quantize_a8_ste(x: torch.Tensor) -> torch.Tensor:
    """Cuantización simétrica INT8 por token con STE (idéntica a trace_quantize_rows_int8)."""
    scale = (x.detach().float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-5) / 127.0).to(x.dtype)
    x_q = torch.clamp(torch.round(x / scale), -127.0, 127.0) * scale
    return x + (x_q - x).detach()


def enable_bitnet_qat_ste_on_lora(model: nn.Module) -> int:
    """
    Aplica QAT-STE W1.58A8 sobre el peso efectivo `W_base + lora_B @ lora_A * scale` de cada capa
    LoRA (PEFT) durante el forward, garantizando 0 choque de cuantización al exportar con `import_bitnet.py`.
    Respeta `disable_adapters` (referencia KL de GRPO) y no rompe `merge_and_unload()`.
    """
    patched = 0
    for mod in model.modules():
        if not (hasattr(mod, "base_layer") and hasattr(mod, "lora_A") and hasattr(mod, "lora_B")):
            continue
        if not isinstance(mod.base_layer, nn.Linear):
            continue

        def _qat_lora_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:  # noqa: ANN001
            w_eff = self.base_layer.weight
            active = getattr(self, "active_adapters", ["default"])
            disabled = getattr(self, "disable_adapters", False)
            merged = getattr(self, "merged", False)
            if not disabled and not merged:
                for adapter in active:
                    if adapter in self.lora_A and adapter in self.lora_B:
                        la = self.lora_A[adapter].weight
                        lb = self.lora_B[adapter].weight
                        sc = self.scaling[adapter]
                        delta = (lb @ la) * sc
                        w_eff = w_eff + delta.to(w_eff.dtype)
            w_q = quantize_w158_ste(w_eff)
            x_q = quantize_a8_ste(x)
            bias = self.base_layer.bias
            return F.linear(x_q, w_q.to(x_q.dtype), bias.to(x_q.dtype) if bias is not None else None)

        mod.forward = types.MethodType(_qat_lora_forward, mod)
        patched += 1
    return patched


def sft_loss_on_completion(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_lengths: torch.Tensor,
    pad_id: int = 0,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Cross-entropy causal calculada estrictamente sobre los tokens del completado
    (enmascarando prompt, padding y opcionalmente spans como `<tool_response>`).

    Args:
        logits: [B, T, V]
        input_ids: [B, T]
        prompt_lengths: [B] longitud de cada prompt (>= 1)
        loss_mask: [B, T] opcional (1 = incluir en pérdida, 0 = ignorar). Si se pasa,
                   se respeta explícitamente sin enmascarar el EOS final aunque pad_id == eos_id.
    """
    bsz, seq_len, vocab = logits.shape
    if seq_len < 2:
        raise ValueError("seq_len debe ser >= 2 para calcular pérdida causal SFT")

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:]

    pos = torch.arange(1, seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
    prompt_mask = pos < prompt_lengths.unsqueeze(1)
    if loss_mask is not None:
        ignore_mask = prompt_mask | (loss_mask[:, 1:] <= 0)
    else:
        ignore_mask = prompt_mask | (shift_labels == pad_id)
    shift_labels = shift_labels.masked_fill(ignore_mask, -100)

    per_tok = F.cross_entropy(
        shift_logits.reshape(-1, vocab),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    )
    valid_mask = (shift_labels.reshape(-1) != -100).to(per_tok.dtype)
    return (per_tok * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)


def gather_token_logprobs(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Extrae log p(target_ids | contexto) con shape [B, T-1] sin materializar [B, T-1, V] FP32."""
    shift_logits = logits[:, :-1, :]
    bsz, seq_len_m1, vocab = shift_logits.shape
    shift_targets = target_ids[:, 1:]
    nll = F.cross_entropy(
        shift_logits.float().reshape(-1, vocab),
        shift_targets.reshape(-1),
        reduction="none",
    )
    return (-nll).reshape(bsz, seq_len_m1)


def grpo_policy_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    log_probs_ref: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
    defer_metrics: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Objetivo GRPO / RLVR (sin red crítico):
      L = -mean( min(r * A, clip(r, 1-eps, 1+eps) * A) - kl_beta * KL_schulman )
    donde KL_schulman = exp(log_ref - log_new) - (log_ref - log_new) - 1 >= 0.

    Si `defer_metrics=True`, devuelve tensores escalares desconectados sin llamar `.item()`
    para evitar sincronizaciones host-device antes de `loss.backward()` en TPU/XLA.
    """
    mask = completion_mask.float()
    denom = mask.sum().clamp_min(1.0)

    log_ratio = (log_probs_new - log_probs_old).clamp(-20.0, 20.0)
    ratio = torch.exp(log_ratio)
    adv = advantages.unsqueeze(1)

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_obj = torch.minimum(surr1, surr2)

    log_ref_diff = (log_probs_ref - log_probs_new).clamp(-20.0, 20.0)
    kl_per_tok = torch.exp(log_ref_diff) - log_ref_diff - 1.0

    per_tok_loss = -(policy_obj - kl_beta * kl_per_tok)
    loss = (per_tok_loss * mask).sum() / denom

    kl_mean_t = ((kl_per_tok * mask).sum() / denom).detach()
    ratio_mean_t = ((ratio * mask).sum() / denom).detach()
    if defer_metrics:
        metrics: dict[str, Any] = {
            "loss": loss.detach(),
            "kl_mean": kl_mean_t,
            "ratio_mean": ratio_mean_t,
        }
        return loss, metrics

    metrics = {
        "loss": float(loss.detach().item()),
        "kl_mean": float(kl_mean_t.item()),
        "ratio_mean": float(ratio_mean_t.item()),
    }
    return loss, metrics


def score_group_completions(
    completions: list[str],
    expected_answer: str,
    *,
    max_target_chars: int = 2048,
) -> tuple[list[float], list[float]]:
    """Evalúa un grupo de G completados con recompensas RLVR y devuelve (rewards, advantages)."""
    rewards = [
        score_reasoning_completion(c, expected_answer, max_target_chars=max_target_chars).total_reward
        for c in completions
    ]
    advantages = compute_grpo_advantages(rewards)
    return rewards, advantages
