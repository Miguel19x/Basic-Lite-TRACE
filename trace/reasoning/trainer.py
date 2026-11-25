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

import torch
import torch.nn.functional as F

from trace.reasoning.rewards import compute_grpo_advantages, score_reasoning_completion


def sft_loss_on_completion(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_lengths: torch.Tensor,
    pad_id: int = 0,
) -> torch.Tensor:
    """
    Cross-entropy causal calculada estrictamente sobre los tokens del completado
    (enmascarando el prompt y el padding con ignore_index=-100).

    Args:
        logits: [B, T, V]
        input_ids: [B, T]
        prompt_lengths: [B] longitud de cada prompt (>= 1)
    """
    bsz, seq_len, vocab = logits.shape
    if seq_len < 2:
        raise ValueError("seq_len debe ser >= 2 para calcular pérdida causal SFT")

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].clone()

    # Enmascarar posiciones que pertenecen al prompt (target index < prompt_len) o padding
    pos = torch.arange(1, seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
    prompt_mask = pos < prompt_lengths.unsqueeze(1)
    pad_mask = shift_labels == pad_id
    shift_labels[prompt_mask | pad_mask] = -100

    valid_tokens = int((shift_labels != -100).sum().item())
    if valid_tokens == 0:
        return shift_logits.sum() * 0.0

    return F.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def gather_token_logprobs(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Extrae log p(target_ids | contexto) con shape [B, T-1]."""
    log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    targets = target_ids[:, 1:].unsqueeze(-1)
    return log_probs.gather(dim=-1, index=targets).squeeze(-1)


def grpo_policy_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    log_probs_ref: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Objetivo GRPO / RLVR (sin red crítico):
      L = -mean( min(r * A, clip(r, 1-eps, 1+eps) * A) - kl_beta * KL_schulman )
    donde KL_schulman = exp(log_ref - log_new) - (log_ref - log_new) - 1 >= 0.
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

    metrics = {
        "loss": float(loss.detach().item()),
        "kl_mean": float(((kl_per_tok * mask).sum() / denom).detach().item()),
        "ratio_mean": float(((ratio * mask).sum() / denom).detach().item()),
    }
    return loss, metrics


def score_group_completions(
    completions: list[str],
    expected_answer: str,
    *,
    max_target_chars: int = 640,
) -> tuple[list[float], list[float]]:
    """Evalúa un grupo de G completados con recompensas RLVR y devuelve (rewards, advantages)."""
    rewards = [
        score_reasoning_completion(c, expected_answer, max_target_chars=max_target_chars).total_reward
        for c in completions
    ]
    advantages = compute_grpo_advantages(rewards)
    return rewards, advantages
