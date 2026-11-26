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

from trace.eval.bench_4domains import _extract_json_object
from trace.lm.config import TRACELMConfig
from trace.lm.gqa import BitNetGQALM
from trace.reasoning.dataset import (
    ReasoningTraceExample,
    build_verified_tir_dataset,
    format_llama3_chat_prompt,
)
from trace.reasoning.trainer import (
    enable_bitnet_qat_ste_on_lora,
    gather_token_logprobs,
    grpo_policy_loss,
    quantize_w158_ste,
    score_group_completions,
    sft_loss_on_completion,
)
from trace.tir.tools import TIRToolRegistry


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
    grpo_steps: int,
    grpo_group_size: int,
    grpo_max_new_tokens: int,
    grad_accum: int,
    lr: float,
    max_seq_len: int,
    use_lora: bool,
    qat_ste: bool,
    lora_r: int,
    lora_alpha: int,
) -> dict[str, Any]:
    """
    Entrenamiento SFT <think> + TIR + GRPO corto sobre HF 2B (`microsoft/bitnet-b1.58-2B-4T-bf16`).
    Con `--use-lora` (PEFT), usa `disable_adapter()` como política de referencia KL para evitar
    duplicar el modelo 2B en VRAM (cabe holgado en una sola GPU T4 de 16 GB en Kaggle).
    """
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

    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise RuntimeError("`--use-lora` requiere instalar `peft` (pip install peft).") from exc
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_cfg)
        if qat_ste:
            n_patched = enable_bitnet_qat_ste_on_lora(model)
            print(f"[QAT-STE] Activada cuantizacion W1.58A8 STE en {n_patched} capas LoRA.", flush=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # En FP16 + GradScaler (T4), los parámetros entrenables deben vivir en FP32 para evitar
    # `ValueError: Attempting to unscale FP16 gradients`.
    use_scaler = device == "cuda" and dtype == torch.float16
    if use_scaler:
        for p in trainable_params:
            p.data = p.data.float()

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    stop_id = int(eot_id) if isinstance(eot_id, int) and eot_id >= 0 else int(tokenizer.eos_token_id)

    def _encode_sft_with_tool_response_mask(ex: ReasoningTraceExample) -> tuple[list[int], int, list[int]]:
        """Codifica con plantilla LLaMA-3 y enmascara `<tool_response>...</tool_response>` a 0."""
        chat_prompt = format_llama3_chat_prompt(ex.prompt)
        p_ids = tokenizer.encode(chat_prompt, add_special_tokens=False)
        target = ex.formatted_target
        resp_start = target.find("<tool_response>")
        resp_end = target.find("</tool_response>")
        comp_ids: list[int] = []
        comp_mask: list[int] = []
        if resp_start != -1 and resp_end != -1 and resp_end > resp_start:
            end_tag = resp_end + len("</tool_response>")
            before = tokenizer.encode(target[:resp_start], add_special_tokens=False)
            middle = tokenizer.encode(target[resp_start:end_tag], add_special_tokens=False)
            after = tokenizer.encode(target[end_tag:], add_special_tokens=False)
            comp_ids = before + middle + after + [stop_id]
            comp_mask = ([1] * len(before)) + ([0] * len(middle)) + ([1] * (len(after) + 1))
        else:
            body = tokenizer.encode(target, add_special_tokens=False) + [stop_id]
            comp_ids = body
            comp_mask = [1] * len(body)

        ids = (p_ids + comp_ids)[:max_seq_len]
        mask = ([0] * len(p_ids) + comp_mask)[:max_seq_len]
        return ids, min(len(p_ids), len(ids) - 1), mask

    # 1. Fase SFT enmascarada (sin entrenar sobre `<tool_response>` y sin enmascarar `stop_id`)
    losses: list[float] = []
    model.train()
    optimizer.zero_grad()
    eff_accum = max(1, grad_accum)

    for step in range(sft_steps):
        ex = examples[step % len(examples)]
        ids, p_len, mask_list = _encode_sft_with_tool_response_mask(ex)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        pl = torch.tensor([p_len], dtype=torch.long, device=device)
        lmask = torch.tensor([mask_list], dtype=torch.float32, device=device)

        with torch.amp.autocast("cuda", dtype=dtype, enabled=(device == "cuda")):
            out = model(input_ids=x)
            raw_loss = sft_loss_on_completion(out.logits, x, pl, pad_id=int(tokenizer.pad_token_id), loss_mask=lmask)
            loss = raw_loss / eff_accum

        scaler.scale(loss).backward()
        losses.append(float(raw_loss.detach().item()))

        if (step + 1) % eff_accum == 0 or (step + 1) == sft_steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        if (step + 1) % 100 == 0 or (step + 1) == sft_steps:
            print(f"[SFT {step + 1}/{sft_steps}] loss={losses[-1]:.4f}", flush=True)

    # 2. Fase GRPO / RLVR corta verificable (con ejecución real de `calc` si emite `<tool_call>`)
    grpo_metrics: list[dict[str, float]] = []
    reg_calc = TIRToolRegistry()
    tir_examples = [ex for ex in examples if ex.category != "general_no_tool"]
    if grpo_steps > 0 and use_lora and tir_examples:
        grpo_lr = lr * 0.25
        for pg in optimizer.param_groups:
            pg["lr"] = grpo_lr

        for step in range(grpo_steps):
            ex = tir_examples[step % len(tir_examples)]
            p_ids = tokenizer.encode(format_llama3_chat_prompt(ex.prompt), add_special_tokens=False)
            prompt_tensor = torch.tensor([p_ids], dtype=torch.long, device=device)
            p_len = len(p_ids)

            model.eval()
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
            model.config.use_cache = True
            with torch.no_grad():
                gen_out = model.generate(
                    input_ids=prompt_tensor.expand(grpo_group_size, -1),
                    max_new_tokens=grpo_max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=int(tokenizer.pad_token_id),
                    eos_token_id=stop_id,
                )
            model.config.use_cache = False
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()

            completions = []
            for g in range(grpo_group_size):
                comp = tokenizer.decode(gen_out[g, p_len:], skip_special_tokens=True)
                # Si la traza invocó `calc` con JSON válido, ejecutarlo realmente y adjuntar `<answer>`
                parsed = _extract_json_object(comp)
                if parsed and parsed.get("tool") == "calc" and isinstance(parsed.get("args"), dict):
                    res = reg_calc.dispatch("calc", parsed["args"])
                    if res.get("ok") and "<answer>" not in comp:
                        comp = f"{comp}\n<answer>{res['result']}</answer>"
                completions.append(comp)

            rewards, adv_list = score_group_completions(completions, ex.answer)
            if all(a == 0.0 for a in adv_list):
                grpo_metrics.append({"loss": 0.0, "kl_mean": 0.0, "ratio_mean": 1.0, "mean_reward": sum(rewards) / len(rewards)})
                continue

            # Construir máscara de completado preservando el primer `stop_id` (no enmascarar EOS real)
            comp_mask = torch.zeros((grpo_group_size, gen_out.shape[1] - 1), dtype=torch.float32, device=device)
            for g in range(grpo_group_size):
                seq_g = gen_out[g].tolist()
                end_pos = len(seq_g) - 1
                for idx in range(p_len, len(seq_g)):
                    if seq_g[idx] == stop_id:
                        end_pos = idx
                        break
                comp_mask[g, max(0, p_len - 1) : end_pos] = 1.0

            adv = torch.tensor(adv_list, dtype=torch.float32, device=device)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype, enabled=(device == "cuda")):
                with model.disable_adapter():
                    ref_logits = model(input_ids=gen_out).logits
                    log_ref = gather_token_logprobs(ref_logits, gen_out)
                old_logits = model(input_ids=gen_out).logits
                log_old = gather_token_logprobs(old_logits, gen_out)

            model.train()
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=dtype, enabled=(device == "cuda")):
                new_logits = model(input_ids=gen_out).logits
                log_new = gather_token_logprobs(new_logits, gen_out)
                loss, m = grpo_policy_loss(log_new, log_old, log_ref, adv, comp_mask, kl_beta=0.04)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            m["mean_reward"] = sum(rewards) / len(rewards)
            grpo_metrics.append(m)
            if (step + 1) % 8 == 0 or (step + 1) == grpo_steps:
                print(f"[GRPO {step + 1}/{grpo_steps}] reward={m['mean_reward']:.3f} kl={m['kl_mean']:.4f}", flush=True)

    if use_lora:
        model = model.merge_and_unload()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Verificacion rapida post-cuantizacion ternaria antes de descargar el .bin de Kaggle
    sanity_sample = ""
    try:
        model.eval()
        model.config.use_cache = True
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        with torch.no_grad():
            for name, mod in model.named_modules():
                if isinstance(mod, torch.nn.Linear) and "lm_head" not in name:
                    mod.weight.data.copy_(quantize_w158_ste(mod.weight.data))
            test_prompt = format_llama3_chat_prompt(
                "Calcula el valor entero exacto de `(12 * 13) + 4`. Responde únicamente con el número entero."
            )
            in_ids = tokenizer.encode(test_prompt, add_special_tokens=False)
            in_t = torch.tensor([in_ids], dtype=torch.long, device=device)
            out_t = model.generate(
                input_ids=in_t,
                max_new_tokens=96,
                do_sample=False,
                pad_token_id=int(tokenizer.pad_token_id),
                eos_token_id=stop_id,
            )
            sanity_sample = tokenizer.decode(out_t[0, len(in_ids) :], skip_special_tokens=True)
            print(f"[TERNARY SANITY CHECK]\n{sanity_sample}", flush=True)
    except Exception as exc:
        sanity_sample = f"sanity_error: {exc}"

    return {
        "backend": "hf-2b",
        "model_id": model_id,
        "device": device,
        "use_lora": use_lora,
        "qat_ste": qat_ste,
        "sft_steps": sft_steps,
        "sft_loss_start": losses[0] if losses else 0.0,
        "sft_loss_end": losses[-1] if losses else 0.0,
        "grpo_steps": len(grpo_metrics),
        "grpo_last_metrics": grpo_metrics[-1] if grpo_metrics else {},
        "ternary_sanity_sample": sanity_sample,
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
    parser.add_argument("--grpo-group-size", type=int, default=4)
    parser.add_argument("--grpo-max-new-tokens", type=int, default=96)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-id", default="microsoft/bitnet-b1.58-2B-4T-bf16")
    parser.add_argument("--out-dir", default="/tmp/trace_phase5_out")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--use-lora", action="store_true", help="Activa LoRA (PEFT) + merge al guardar")
    parser.add_argument("--qat-ste", action="store_true", help="Aplica QAT-STE W1.58A8 en capas LoRA (obligatorio para BitNet)")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
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
            grpo_steps=args.grpo_steps,
            grpo_group_size=args.grpo_group_size,
            grpo_max_new_tokens=args.grpo_max_new_tokens,
            grad_accum=args.grad_accum,
            lr=args.lr,
            max_seq_len=args.max_seq_len,
            use_lora=args.use_lora,
            qat_ste=args.qat_ste,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
        )

    summary["elapsed_sec"] = round(time.perf_counter() - t0, 3)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
