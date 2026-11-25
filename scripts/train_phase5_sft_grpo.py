#!/usr/bin/env python3
# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Pipeline de Fase 5: Destilación SFT (<think> + TIR) + GRPO / RLVR corto verificable.

Modos soportados:
  1. `--export-dataset <ruta.jsonl>`: genera y exporta trazas SFT verificadas en sandbox (>=10% datos generales, 0 fuga 4B).
  2. `--backend mini-gqa`: piloto local determinista en CPU (2 hilos) sobre `BitNetGQALM` compacto para validar SFT + GRPO.
  3. `--backend hf-2b`: entrenamiento SFT + GRPO corto en GPU (ej. Kaggle 2x T4) sobre `microsoft/bitnet-b1.58-2B-4T`,
     guardando checkpoint listo para convertir a `Paged BitCOS v1.1` con `tools/import_bitnet.py`.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from trace.lm.config import TRACELMConfig
from trace.lm.gqa import BitNetGQALM
from trace.reasoning.dataset import ReasoningTraceExample, build_verified_tir_dataset
from trace.reasoning.trainer import (
    gather_token_logprobs,
    grpo_policy_loss,
    score_group_completions,
    sft_loss_on_completion,
)


def export_dataset_jsonl(examples: list[ReasoningTraceExample], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")


def _encode_bytes_pair(prompt: str, target: str, max_len: int) -> tuple[list[int], int]:
    """Codificación determinista byte-level (0..255, BOS=258, EOS=259) para piloto `mini-gqa`."""
    p_ids = [258] + list(prompt.encode("utf-8"))
    t_ids = list(target.encode("utf-8")) + [259]
    if len(p_ids) >= max_len - 4:
        p_ids = p_ids[: max_len - 4]
    ids = (p_ids + t_ids)[:max_len]
    return ids, len(p_ids)


def run_mini_gqa_pilot(
    examples: list[ReasoningTraceExample],
    *,
    sft_steps: int = 20,
    grpo_steps: int = 4,
    lr: float = 3e-3,
    seed: int = 42,
) -> dict[str, Any]:
    """Piloto local reproducible en CPU (2 hilos) que valida SFT enmascarado + GRPO."""
    torch.set_num_threads(2)
    torch.manual_seed(seed)

    cfg = TRACELMConfig(
        vocab_size=264,
        max_seq_len=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        attn_type="gqa",
        use_gated_ffn=True,
        tie_word_embeddings=True,
    )
    model = BitNetGQALM(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    encoded = [_encode_bytes_pair(ex.prompt, ex.formatted_target, max_len=cfg.max_seq_len) for ex in examples]
    sft_losses: list[float] = []
    model.train()

    for step in range(sft_steps):
        ids, p_len = encoded[step % len(encoded)]
        x = torch.tensor([ids], dtype=torch.long)
        pl = torch.tensor([p_len], dtype=torch.long)
        optimizer.zero_grad()
        logits = model(x)[0]
        loss = sft_loss_on_completion(logits, x, pl, pad_id=256)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        sft_losses.append(float(loss.detach().item()))

    # Referencia congelada post-SFT para penalización KL de GRPO
    ref_model = copy.deepcopy(model).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    grpo_metrics: list[dict[str, float]] = []
    for step in range(grpo_steps):
        ex = examples[step % len(examples)]
        # Grupo G=2: completado bueno (traza verificada) vs completado degradado
        completions = [ex.formatted_target, f"<think>adivino</think><answer>{ex.answer}999</answer>"]
        rewards, adv_list = score_group_completions(completions, ex.answer)

        seqs = []
        p_lens = []
        for comp in completions:
            ids, p_len = _encode_bytes_pair(ex.prompt, comp, max_len=cfg.max_seq_len)
            seqs.append(ids)
            p_lens.append(p_len)

        max_t = max(len(s) for s in seqs)
        batch_ids = torch.full((len(seqs), max_t), 256, dtype=torch.long)
        comp_mask = torch.zeros((len(seqs), max_t - 1), dtype=torch.float32)
        for i, (s, p_len) in enumerate(zip(seqs, p_lens)):
            batch_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            comp_mask[i, max(0, p_len - 1) : len(s) - 1] = 1.0

        adv = torch.tensor(adv_list, dtype=torch.float32)
        with torch.no_grad():
            ref_logits = ref_model(batch_ids)[0]
            log_ref = gather_token_logprobs(ref_logits, batch_ids)
            old_logits = model(batch_ids)[0]
            log_old = gather_token_logprobs(old_logits, batch_ids)

        optimizer.zero_grad()
        new_logits = model(batch_ids)[0]
        log_new = gather_token_logprobs(new_logits, batch_ids)
        loss, m = grpo_policy_loss(log_new, log_old, log_ref, adv, comp_mask, kl_beta=0.04)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        m["mean_reward"] = sum(rewards) / len(rewards)
        grpo_metrics.append(m)

    return {
        "backend": "mini-gqa",
        "num_examples": len(examples),
        "sft_steps": sft_steps,
        "sft_loss_start": sft_losses[0] if sft_losses else 0.0,
        "sft_loss_end": sft_losses[-1] if sft_losses else 0.0,
        "grpo_steps": grpo_steps,
        "grpo_last_metrics": grpo_metrics[-1] if grpo_metrics else {},
    }


def run_hf_2b_training(
    examples: list[ReasoningTraceExample],
    *,
    model_id: str,
    out_dir: Path,
    sft_steps: int,
    lr: float,
    max_seq_len: int,
) -> dict[str, Any]:
    """Entrenamiento SFT <think> + TIR sobre HF 2B (pensado para GPU Kaggle 2x T4)."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "El backend 'hf-2b' requiere `transformers` (entorno GPU/Kaggle). "
            "En CPU local usa `--backend mini-gqa` o `--export-dataset`."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    losses: list[float] = []
    model.train()
    for step in range(sft_steps):
        ex = examples[step % len(examples)]
        p_ids = tokenizer.encode(ex.prompt + "\n", add_special_tokens=True)
        t_ids = tokenizer.encode(ex.formatted_target, add_special_tokens=False) + [tokenizer.eos_token_id]
        ids = (p_ids + t_ids)[:max_seq_len]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        pl = torch.tensor([min(len(p_ids), len(ids) - 1)], dtype=torch.long, device=device)

        optimizer.zero_grad()
        out = model(input_ids=x)
        loss = sft_loss_on_completion(out.logits, x, pl, pad_id=int(tokenizer.pad_token_id))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().item()))

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return {
        "backend": "hf-2b",
        "model_id": model_id,
        "device": device,
        "sft_steps": sft_steps,
        "sft_loss_start": losses[0] if losses else 0.0,
        "sft_loss_end": losses[-1] if losses else 0.0,
        "saved_dir": str(out_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 5: SFT <think> + TIR y GRPO verificable")
    parser.add_argument("--backend", choices=["export-only", "mini-gqa", "hf-2b"], default="mini-gqa")
    parser.add_argument("--export-dataset", default="", help="Ruta JSONL para exportar dataset verificado")
    parser.add_argument("--num-math", type=int, default=32)
    parser.add_argument("--num-code", type=int, default=16)
    parser.add_argument("--num-factual", type=int, default=12)
    parser.add_argument("--general-ratio", type=float, default=0.15)
    parser.add_argument("--sft-steps", type=int, default=20)
    parser.add_argument("--grpo-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-id", default="microsoft/bitnet-b1.58-2B-4T")
    parser.add_argument("--out-dir", default="/tmp/trace_phase5_out")
    parser.add_argument("--max-seq-len", type=int, default=512)
    args = parser.parse_args()

    t0 = time.perf_counter()
    examples = build_verified_tir_dataset(
        num_math=args.num_math,
        num_code=args.num_code,
        num_factual=args.num_factual,
        general_ratio=args.general_ratio,
        seed=args.seed,
    )
    if args.export_dataset:
        export_dataset_jsonl(examples, Path(args.export_dataset))

    if args.backend == "export-only":
        summary: dict[str, Any] = {
            "backend": "export-only",
            "num_examples": len(examples),
            "exported": args.export_dataset,
        }
    elif args.backend == "mini-gqa":
        summary = run_mini_gqa_pilot(
            examples,
            sft_steps=args.sft_steps,
            grpo_steps=args.grpo_steps,
            lr=args.lr,
            seed=args.seed,
        )
    else:
        summary = run_hf_2b_training(
            examples,
            model_id=args.model_id,
            out_dir=Path(args.out_dir),
            sft_steps=args.sft_steps,
            lr=args.lr,
            max_seq_len=args.max_seq_len,
        )

    summary["elapsed_sec"] = round(time.perf_counter() - t0, 3)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
