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
import math
import time
from collections.abc import Callable
from contextlib import nullcontext
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


def _try_import_xla() -> Any | None:
    """Importa `torch_xla.core.xla_model` si está disponible en el entorno (TPU v5e-8)."""
    try:
        import torch_xla.core.xla_model as xm  # type: ignore[import-not-found]
    except ImportError:
        return None
    return xm


def resolve_device_and_dtype(
    requested_device: str = "auto",
    *,
    xla_available: bool | None = None,
    cuda_available: bool | None = None,
    cuda_bf16_supported: bool | None = None,
    xla_device_fn: Callable[[], Any] | None = None,
) -> tuple[Any, str, torch.dtype]:
    """
    Resuelve dispositivo, backend (`xla`/`cuda`/`cpu`) y precisión nativa:
    - `xla` (TPU v5e-8): `torch.bfloat16` nativo.
    - `cuda` con BF16 nativo: `torch.bfloat16`.
    - `cuda` sin BF16 nativo (ej. T4): `torch.float32` (evita overflow FP16 de BitNet b1.58).
    - `cpu`: `torch.float32`.
    """
    xm = _try_import_xla() if xla_available is None else None
    has_xla = (xm is not None) if xla_available is None else xla_available
    dev_fn = xla_device_fn if xla_device_fn is not None else (xm.xla_device if xm is not None else None)
    has_cuda = torch.cuda.is_available() if cuda_available is None else cuda_available
    has_cuda_bf16 = (
        (torch.cuda.is_bf16_supported() if has_cuda else False)
        if cuda_bf16_supported is None
        else cuda_bf16_supported
    )

    if requested_device == "xla":
        if not has_xla or dev_fn is None:
            raise RuntimeError(
                "Se solicitó `--device xla`, pero `torch_xla` no está instalado o no hay dispositivo TPU."
            )
        return dev_fn(), "xla", torch.bfloat16

    if requested_device == "auto":
        if has_xla and dev_fn is not None:
            if xla_available is None:
                try:
                    return dev_fn(), "xla", torch.bfloat16
                except Exception:
                    pass
            else:
                return dev_fn(), "xla", torch.bfloat16
        if has_cuda:
            return "cuda", "cuda", (torch.bfloat16 if has_cuda_bf16 else torch.float32)
        return "cpu", "cpu", torch.float32

    if requested_device == "cuda":
        if not has_cuda:
            raise RuntimeError("Se solicitó `--device cuda`, pero CUDA no está disponible.")
        return "cuda", "cuda", (torch.bfloat16 if has_cuda_bf16 else torch.float32)

    if requested_device == "cpu":
        return "cpu", "cpu", torch.float32

    raise ValueError(f"Dispositivo no soportado: {requested_device!r}")


def pad_batch_to_static_length(
    sequences: list[list[int]],
    masks: list[list[int]] | None = None,
    *,
    max_length: int,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Aplica padding/truncado a longitud estática fija `max_length` para evitar
    recompilaciones de grafos HLO en TPU/XLA.
    """
    if max_length < 2:
        raise ValueError(f"max_length debe ser >= 2, recibido {max_length}")
    if not sequences:
        raise ValueError("sequences no puede estar vacío")
    if masks is not None and len(masks) != len(sequences):
        raise ValueError("masks debe tener la misma longitud que sequences")

    batch_size = len(sequences)
    ids_out = torch.full((batch_size, max_length), int(pad_id), dtype=torch.long)
    mask_out = torch.zeros((batch_size, max_length), dtype=torch.float32)

    for i, seq in enumerate(sequences):
        n = min(len(seq), max_length)
        if n > 0:
            ids_out[i, :n] = torch.tensor(seq[:n], dtype=torch.long)
            if masks is not None:
                m_row = masks[i][:n]
                mask_out[i, : len(m_row)] = torch.tensor(m_row, dtype=torch.float32)
            else:
                mask_out[i, :n] = 1.0

    return ids_out, mask_out


def save_xla_or_torch_checkpoint(
    model: Any,
    path: Path,
    *,
    backend: str,
    xm: Any | None = None,
) -> None:
    """Guarda `state_dict` usando `xm.save` en XLA o `torch.save` en CPU/CUDA."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if backend == "xla" and xm is not None:
        xm.save(model.state_dict(), str(path))
    else:
        torch.save(model.state_dict(), path)


def run_sft_loop(
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    trainable_params: list[torch.nn.Parameter],
    step_loss_fn: Callable[[int], Any],
    sft_steps: int,
    grad_accum: int,
    lr: float,
    backend: str,
    scaler: Any | None = None,
    xm: Any | None = None,
    log_every: int = 80,
    max_wall_hours: float = 8.0,
    clock_fn: Callable[[], float] = time.monotonic,
    start_sec: float | None = None,
    checkpoint_path: Path | None = None,
    early_guardrail: bool = False,
) -> dict[str, Any]:
    """
    Ejecuta el bucle SFT con acumulación de gradiente, decaimiento coseno,
    sincronización XLA diferida (sin `.item()` por paso en TPU) y guardia de tiempo de muro.
    """
    is_xla = backend == "xla"
    if is_xla and xm is None:
        raise ValueError("backend='xla' requiere pasar el módulo `xm` de torch_xla.")

    eff_accum = max(1, grad_accum)
    total_opt_steps = max(1, (sft_steps + eff_accum - 1) // eff_accum)
    wall_limit_sec = max_wall_hours * 3600.0 if max_wall_hours > 0.0 else 0.0
    wall_start_sec = clock_fn() if start_sec is None else start_sec

    losses: list[float] = []
    opt_step = 0
    steps_completed = 0
    stopped_early_wall_budget = False
    log_interval = max(1, log_every)
    log_win_sum: torch.Tensor | None = None
    log_win_count = 0
    win_start_sum: torch.Tensor | None = None
    win_end_sum: torch.Tensor | None = None
    last_cur_lr = float(lr)

    if is_xla:
        lr_device = trainable_params[0].device if trainable_params else torch.device("cpu")
        for pg in optimizer.param_groups:
            if isinstance(pg["lr"], torch.Tensor):
                pg["lr"] = pg["lr"].to(device=lr_device, dtype=torch.float32)
            else:
                pg["lr"] = torch.tensor(float(pg["lr"]), dtype=torch.float32, device=lr_device)

    model.train()
    optimizer.zero_grad()

    for step in range(sft_steps):
        raw_loss = step_loss_fn(step)
        loss = raw_loss / eff_accum

        if is_xla:
            raw_det = raw_loss.detach()
            log_win_sum = raw_det if log_win_sum is None else (log_win_sum + raw_det)
            log_win_count += 1
            if early_guardrail and sft_steps >= 240:
                if step < 40:
                    win_start_sum = raw_det if win_start_sum is None else (win_start_sum + raw_det)
                elif 120 <= step < 160:
                    win_end_sum = raw_det if win_end_sum is None else (win_end_sum + raw_det)
        else:
            raw_val = float(raw_loss.detach().item())
            if not math.isfinite(raw_val):
                raise RuntimeError(f"[ABORTADO] Pérdida SFT no finita ({raw_val}) en paso {step + 1}.")
            losses.append(raw_val)

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        is_opt_step = ((step + 1) % eff_accum == 0) or ((step + 1) == sft_steps)
        if is_opt_step:
            opt_step += 1
            progress = min(1.0, opt_step / total_opt_steps)
            cur_lr = lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
            last_cur_lr = float(cur_lr)
            for pg in optimizer.param_groups:
                if is_xla and isinstance(pg["lr"], torch.Tensor):
                    pg["lr"].fill_(last_cur_lr)
                else:
                    pg["lr"] = cur_lr
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            if is_xla and xm is not None:
                xm.optimizer_step(optimizer, barrier=True)
            elif scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        elif is_xla and xm is not None:
            xm.mark_step()

        steps_completed = step + 1
        wall_hit = wall_limit_sec > 0.0 and (clock_fn() - wall_start_sec) >= wall_limit_sec
        is_log_step = (steps_completed % log_interval == 0) or (steps_completed == sft_steps)
        is_guard_step = early_guardrail and (steps_completed == 160) and sft_steps >= 240

        if is_xla and (is_log_step or wall_hit) and log_win_sum is not None and log_win_count > 0:
            avg_val = float((log_win_sum / log_win_count).item())
            if not math.isfinite(avg_val):
                raise RuntimeError(f"[ABORTADO] Pérdida SFT no finita ({avg_val}) en paso {steps_completed}.")
            losses.append(avg_val)
            log_win_sum = None
            log_win_count = 0

        if is_log_step and losses:
            if is_xla:
                avg_recent = losses[-1]
            else:
                window = min(log_interval, len(losses))
                avg_recent = sum(losses[-window:]) / window
            print(
                f"[SFT {steps_completed}/{sft_steps}] avg_loss={avg_recent:.4f} lr={last_cur_lr:.2e}",
                flush=True,
            )

        if is_guard_step:
            if not is_xla and len(losses) >= 160:
                early_avg = sum(losses[:40]) / 40.0
                mid_avg = sum(losses[120:160]) / 40.0
                if mid_avg > early_avg * 0.85:
                    raise RuntimeError(
                        f"[ABORTADO TEMPRANO EN PASO 160] La pérdida no desciende "
                        f"(inicio={early_avg:.4f} -> paso160={mid_avg:.4f}). "
                        "Revisa --lr antes de gastar el resto de la sesión."
                    )
            elif is_xla and win_start_sum is not None and win_end_sum is not None:
                early_avg = float((win_start_sum / 40.0).item())
                mid_avg = float((win_end_sum / 40.0).item())
                if not (math.isfinite(early_avg) and math.isfinite(mid_avg)):
                    raise RuntimeError(
                        f"[ABORTADO] Pérdida SFT XLA no finita en guardrail paso 160 "
                        f"(inicio={early_avg}, paso160={mid_avg})."
                    )
                if mid_avg > early_avg * 0.85:
                    raise RuntimeError(
                        f"[ABORTADO TEMPRANO EN PASO 160] La pérdida XLA no desciende "
                        f"(inicio={early_avg:.4f} -> paso160={mid_avg:.4f}). "
                        "Revisa --lr antes de gastar el resto de la sesión TPU."
                    )

        if wall_hit:
            stopped_early_wall_budget = True
            if checkpoint_path is not None:
                save_xla_or_torch_checkpoint(model, checkpoint_path, backend=backend, xm=xm)
            print(
                f"[WALL-GUARD] Alcanzado límite de {max_wall_hours:.2f}h en paso {steps_completed}/{sft_steps}. "
                "Guardando checkpoint limpio.",
                flush=True,
            )
            break

    return {
        "losses": losses,
        "steps_completed": steps_completed,
        "stopped_early_wall_budget": stopped_early_wall_budget,
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
    requested_device: str = "auto",
    max_wall_hours: float = 8.0,
) -> dict[str, Any]:
    """
    Entrenamiento SFT <think> + TIR + GRPO corto sobre HF 2B (`microsoft/bitnet-b1.58-2B-4T-bf16`).
    Soporta TPU v5e-8 (`--device xla`, BF16 nativo), GPU CUDA (BF16/FP32) y CPU (FP32).
    Con `--use-lora` (PEFT), usa `disable_adapter()` como política de referencia KL para evitar
    duplicar el modelo 2B en memoria.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "El backend 'hf-2b' requiere `transformers` (entorno GPU/TPU/Kaggle). "
            "En CPU local usa `--backend mini-gqa` o `--export-dataset`."
        ) from exc

    device, backend, dtype = resolve_device_and_dtype(requested_device)
    xm = _try_import_xla() if backend == "xla" else None
    is_xla = backend == "xla"
    t_wall_start = time.monotonic()
    wall_limit_sec = max_wall_hours * 3600.0 if max_wall_hours > 0.0 else 0.0

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

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
            if n_patched == 0:
                raise RuntimeError(
                    "[ABORTADO] `--qat-ste` está activado pero `enable_bitnet_qat_ste_on_lora` parcheó 0 capas LoRA."
                )
            print(f"[QAT-STE] Activada cuantizacion W1.58A8 STE en {n_patched} capas LoRA.", flush=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    use_cuda_autocast = backend == "cuda" and dtype == torch.bfloat16
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False) if backend == "cuda" else None

    def _autocast_ctx() -> Any:
        if use_cuda_autocast:
            return torch.amp.autocast("cuda", dtype=dtype, enabled=True)
        return nullcontext()

    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    stop_id = int(eot_id) if isinstance(eot_id, int) and eot_id >= 0 else int(tokenizer.eos_token_id)

    def _run_generation_check(tag: str, prompt_text: str, max_toks: int = 48) -> str:
        model.eval()
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        with torch.no_grad():
            in_ids = tokenizer.encode(format_llama3_chat_prompt(prompt_text), add_special_tokens=False)
            if is_xla:
                # En TPU/XLA evitamos `model.generate()` dinámico en preflight: un único forward
                # estático (`max_seq_len`) compila el grafo HLO 1 sola vez y verifica finitud/argmax.
                model.config.use_cache = False
                in_cpu, _ = pad_batch_to_static_length(
                    [in_ids],
                    max_length=max_seq_len,
                    pad_id=int(tokenizer.pad_token_id),
                )
                in_t = in_cpu.to(device)
                out_logits = model(input_ids=in_t).logits
                if xm is not None:
                    xm.mark_step()
                last_pos = max(0, min(len(in_ids), max_seq_len) - 1)
                last_logits = out_logits[0, last_pos]
                if not bool(torch.isfinite(last_logits).all().item()):
                    raise RuntimeError(f"[ABORTADO EN 5s] Logits no finitos en preflight XLA ({tag}).")
                next_tok_id = int(torch.argmax(last_logits).item())
                txt = tokenizer.decode([next_tok_id], skip_special_tokens=True)
            else:
                model.config.use_cache = True
                in_t = torch.tensor([in_ids], dtype=torch.long, device=device)
                out_t = model.generate(
                    input_ids=in_t,
                    max_new_tokens=max_toks,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=int(tokenizer.pad_token_id),
                    eos_token_id=stop_id,
                )
                txt = tokenizer.decode(out_t[0, len(in_ids) :], skip_special_tokens=True)
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model.train()
        print(f"[{tag}] {txt!r}", flush=True)
        return txt

    # GUARDRAIL 1 (Fail-Fast en 5 s): verificar que el modelo parcheado con LoRA+QAT-STE
    # responde coherentemente ANTES de gastar minutos de GPU/TPU.
    pre_txt = _run_generation_check("PREFLIGHT STEP 0", "Calcula 12 * 13. Responde solo el numero.", max_toks=24)
    if not is_xla and "156" not in pre_txt:
        raise RuntimeError(
            f"[ABORTADO EN 5s] El modelo base parcheado falló el preflight (esperaba '156', obtuvo {pre_txt!r}). "
            "No se gastará cómputo en un modelo descalibrado."
        )

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

    # 1. Fase SFT enmascarada con decaimiento coseno + guardrail temprano en paso 160 + guardia de tiempo
    def _sft_step_loss(step: int) -> torch.Tensor:
        ex = examples[step % len(examples)]
        ids, p_len, mask_list = _encode_sft_with_tool_response_mask(ex)
        if is_xla:
            x_cpu, lmask_cpu = pad_batch_to_static_length(
                [ids],
                [mask_list],
                max_length=max_seq_len,
                pad_id=int(tokenizer.pad_token_id),
            )
            x = x_cpu.to(device)
            lmask = lmask_cpu.to(device)
        else:
            x = torch.tensor([ids], dtype=torch.long, device=device)
            lmask = torch.tensor([mask_list], dtype=torch.float32, device=device)
        pl = torch.tensor([p_len], dtype=torch.long, device=device)

        with _autocast_ctx():
            out = model(input_ids=x)
            return sft_loss_on_completion(out.logits, x, pl, pad_id=int(tokenizer.pad_token_id), loss_mask=lmask)

    # `checkpoint_path=None` evita escribir un .pt intermedio redundante de ~4.8 GB en wall-guard,
    # ya que `run_hf_2b_training` procede inmediatamente a `merge_and_unload()` + `save_pretrained(out_dir)`.
    sft_res = run_sft_loop(
        model=model,
        optimizer=optimizer,
        trainable_params=trainable_params,
        step_loss_fn=_sft_step_loss,
        sft_steps=sft_steps,
        grad_accum=grad_accum,
        lr=lr,
        backend=backend,
        scaler=scaler,
        xm=xm,
        log_every=80,
        max_wall_hours=max_wall_hours,
        start_sec=t_wall_start,
        checkpoint_path=None,
        early_guardrail=True,
    )
    losses: list[float] = sft_res["losses"]
    stopped_early_wall_budget: bool = bool(sft_res["stopped_early_wall_budget"])

    # 2. Fase GRPO / RLVR corta verificable (con ejecución real de `calc` si emite `<tool_call>`)
    grpo_metrics: list[dict[str, float]] = []
    reg_calc = TIRToolRegistry()
    tir_examples = [ex for ex in examples if ex.category != "general_no_tool"]
    grpo_static_len = max(2, max_seq_len + grpo_max_new_tokens)
    if not stopped_early_wall_budget and grpo_steps > 0 and use_lora and tir_examples:
        grpo_lr = lr * 0.25
        for pg in optimizer.param_groups:
            if is_xla and isinstance(pg["lr"], torch.Tensor):
                pg["lr"].fill_(float(grpo_lr))
            else:
                pg["lr"] = grpo_lr

        pad_id = int(tokenizer.pad_token_id)
        for step in range(grpo_steps):
            if wall_limit_sec > 0.0 and (time.monotonic() - t_wall_start) >= wall_limit_sec:
                stopped_early_wall_budget = True
                print(
                    f"[WALL-GUARD] Alcanzado límite de {max_wall_hours:.2f}h durante GRPO paso {step}/{grpo_steps}. "
                    "Procediendo a merge_and_unload() y save_pretrained().",
                    flush=True,
                )
                break

            ex = tir_examples[step % len(tir_examples)]
            p_ids = tokenizer.encode(format_llama3_chat_prompt(ex.prompt), add_special_tokens=False)
            if is_xla:
                p_ids = p_ids[:max_seq_len]
                pad_count = max(0, max_seq_len - len(p_ids))
                left_padded_ids = ([pad_id] * pad_count) + p_ids
                left_attn_mask = ([0] * pad_count) + ([1] * len(p_ids))
                prompt_tensor = torch.tensor([left_padded_ids], dtype=torch.long, device=device)
                prompt_attn_mask = torch.tensor([left_attn_mask], dtype=torch.long, device=device)
                p_len = max_seq_len
            else:
                prompt_tensor = torch.tensor([p_ids], dtype=torch.long, device=device)
                prompt_attn_mask = None
                p_len = len(p_ids)

            model.eval()
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
            model.config.use_cache = True
            with torch.no_grad():
                gen_kwargs: dict[str, Any] = {
                    "input_ids": prompt_tensor.expand(grpo_group_size, -1),
                    "max_new_tokens": grpo_max_new_tokens,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "use_cache": True,
                    "pad_token_id": pad_id,
                    "eos_token_id": stop_id,
                }
                if prompt_attn_mask is not None:
                    gen_kwargs["attention_mask"] = prompt_attn_mask.expand(grpo_group_size, -1)
                if is_xla:
                    try:
                        gen_out = model.generate(**gen_kwargs, cache_implementation="static")
                    except TypeError:
                        gen_out = model.generate(**gen_kwargs)
                    if xm is not None:
                        xm.mark_step()
                else:
                    gen_out = model.generate(**gen_kwargs)
            model.config.use_cache = False
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()

            completions = []
            gen_seqs = gen_out.detach().cpu().tolist()
            for g in range(grpo_group_size):
                comp = tokenizer.decode(gen_seqs[g][p_len:], skip_special_tokens=True)
                # Si la traza invocó `calc` con JSON válido, ejecutarlo realmente y adjuntar `<answer>`
                parsed = _extract_json_object(comp)
                if parsed and parsed.get("tool") == "calc" and isinstance(parsed.get("args"), dict):
                    res = reg_calc.dispatch("calc", parsed["args"])
                    if res.get("ok") and "<answer>" not in comp:
                        comp = f"{comp}\n<answer>{res['result']}</answer>"
                completions.append(comp)

            rewards, adv_list = score_group_completions(completions, ex.answer)
            if all(a == 0.0 for a in adv_list):
                grpo_metrics.append(
                    {"loss": 0.0, "kl_mean": 0.0, "ratio_mean": 1.0, "mean_reward": sum(rewards) / len(rewards)}
                )
                continue

            # En TPU/XLA fijamos forma estática [G, max_seq_len + grpo_max_new_tokens] para los
            # 3 forwards + 1 backward de GRPO, evitando recompilaciones HLO por forma dinámica.
            if is_xla:
                gen_pad_cpu, _ = pad_batch_to_static_length(
                    gen_seqs,
                    max_length=grpo_static_len,
                    pad_id=pad_id,
                )
                gen_batch = gen_pad_cpu.to(device)
            else:
                gen_batch = gen_out

            # Construir máscara de completado en CPU y transferir en 1 solo `.to(device)` (evita ops dinámicas XLA)
            comp_mask_cpu = torch.zeros((grpo_group_size, gen_batch.shape[1] - 1), dtype=torch.float32, device="cpu")
            for g in range(grpo_group_size):
                seq_g = gen_seqs[g]
                max_valid_end = min(len(seq_g) - 1, gen_batch.shape[1] - 1)
                end_pos = max_valid_end
                for idx in range(p_len, min(len(seq_g), gen_batch.shape[1])):
                    if seq_g[idx] == stop_id:
                        end_pos = idx
                        break
                comp_mask_cpu[g, max(0, p_len - 1) : end_pos] = 1.0
            comp_mask = comp_mask_cpu.to(device)

            adv = torch.tensor(adv_list, dtype=torch.float32, device=device)

            with torch.no_grad(), _autocast_ctx():
                with model.disable_adapter():
                    ref_logits = model(input_ids=gen_batch).logits
                    log_ref = gather_token_logprobs(ref_logits, gen_batch)
                    del ref_logits
                if is_xla and xm is not None:
                    xm.mark_step()
                old_logits = model(input_ids=gen_batch).logits
                log_old = gather_token_logprobs(old_logits, gen_batch)
                del old_logits
            if is_xla and xm is not None:
                xm.mark_step()

            model.train()
            optimizer.zero_grad()
            with _autocast_ctx():
                new_logits = model(input_ids=gen_batch).logits
                log_new = gather_token_logprobs(new_logits, gen_batch)
                loss, m_raw = grpo_policy_loss(
                    log_new,
                    log_old,
                    log_ref,
                    adv,
                    comp_mask,
                    kl_beta=0.04,
                    defer_metrics=is_xla,
                )

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            if is_xla and xm is not None:
                xm.optimizer_step(optimizer, barrier=True)
            elif scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            m: dict[str, float] = {
                k: float(v.item()) if isinstance(v, torch.Tensor) else float(v) for k, v in m_raw.items()
            }
            m["mean_reward"] = sum(rewards) / len(rewards)
            grpo_metrics.append(m)
            if (step + 1) % 8 == 0 or (step + 1) == grpo_steps:
                print(f"[GRPO {step + 1}/{grpo_steps}] reward={m['mean_reward']:.3f} kl={m['kl_mean']:.4f}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    if is_xla and xm is not None:
        xm.mark_step()
        model = model.to("cpu")
        if use_lora:
            model = model.merge_and_unload()
        model.save_pretrained(str(out_dir))
    else:
        if use_lora:
            model = model.merge_and_unload()
        model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Verificacion post-merge SIN doble cuantizacion (AutoBitLinear ya cuantiza en W1.58A8 en forward)
    sanity_sample = ""
    if not is_xla:
        try:
            sanity_sample = _run_generation_check(
                "POST-MERGE TERNARY SANITY CHECK",
                "Calcula el valor entero exacto de `(12 * 13) + 4`. Responde únicamente con el número entero.",
                max_toks=96,
            )
        except Exception as exc:
            sanity_sample = f"sanity_error: {exc}"

    return {
        "backend": "hf-2b",
        "model_id": model_id,
        "device": str(device),
        "backend_device": backend,
        "dtype": str(dtype),
        "use_lora": use_lora,
        "qat_ste": qat_ste,
        "sft_steps": int(sft_res["steps_completed"]),
        "sft_loss_start": losses[0] if losses else 0.0,
        "sft_loss_end": losses[-1] if losses else 0.0,
        "grpo_steps": len(grpo_metrics),
        "grpo_last_metrics": grpo_metrics[-1] if grpo_metrics else {},
        "stopped_early_wall_budget": stopped_early_wall_budget,
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
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "xla"],
        default="auto",
        help="Dispositivo de entrenamiento (`xla` para TPU v5e-8 BF16 nativo, `auto` detecta XLA->CUDA->CPU)",
    )
    parser.add_argument(
        "--max-wall-hours",
        type=float,
        default=8.0,
        help="Límite de horas de reloj antes de guardar checkpoint limpio (Kaggle TPU: 9h máx; <=0 desactiva)",
    )
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
            requested_device=args.device,
            max_wall_hours=args.max_wall_hours,
        )

    summary["elapsed_sec"] = round(time.perf_counter() - t0, 3)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
