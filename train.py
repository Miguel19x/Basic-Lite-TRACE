#!/usr/bin/env python3
# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Basic-Lite 0.2 — Pipeline de Entrenamiento Causal de 2 Etapas
Incorpora Abacus Positional Embeddings, Desacoplamiento Causal (Subtotales -> Totales),
Schedule QAT de 2 Etapas (BitNet b1.58) y Telemetría de Saturación/Flips.
"""

import argparse
import math
import os
import random
import time
from trace.models.causal import TRACECausalModel

import torch

from config import HybridConfig
from data import ProceduralCompositionalDataset
from eval import evaluate_causal


def collate_fn(batch, max_seq_len: int, pad_token_id: int, bos_token_id: int, eos_token_id: int):
    """Collate function que devuelve x_0 y las longitudes de los prompts."""
    x_0_list = []
    plen_list = []
    for prompt_b, target_b, _, _ in batch:
        plen = 1 + len(prompt_b)
        plen_list.append(plen)
        full_seq = [bos_token_id] + list(prompt_b) + list(target_b) + [eos_token_id]
        full_seq = full_seq[:max_seq_len]
        pad_len = max_seq_len - len(full_seq)
        full_seq = full_seq + [pad_token_id] * pad_len
        x_0_list.append(full_seq)
    return torch.tensor(x_0_list, dtype=torch.long), plen_list


def train(
    steps: int = 1000,
    lr: float = 3e-4,
    batch_size: int = 32,
    seed: int = 42,
    device_name: str = "auto",
    eval_interval: int = 200,
    warmup_ratio: float = 0.40,
    warmup_steps: int | None = None,
    lr_warmup_steps: int = 300,
    use_bitnet_schedule: bool = False,
    lr_peak: float = 1.0e-3,
    lr_min: float = 1.5e-4,
    bitlinear_wd_init: float = 0.05,
    checkpoint_out: str = "trace_checkpoint.pt",
):
    torch.manual_seed(seed)
    random.seed(seed)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print("=" * 80)
    print("Basic-Lite TRACE — Entrenamiento Causal de 2 Etapas")
    mode_str = "BitNet b1.58 (2 Etapas + Coseno)" if use_bitnet_schedule else "Estándar (LR Fijo)"
    print(f"Modo: {mode_str} | Dispositivo: {device} | Semilla: {seed} | Pasos: {steps} | Lote: {batch_size}")
    if use_bitnet_schedule:
        print(f"Schedule: LR Peak={lr_peak:.2e} -> Min={lr_min:.2e} | LR Warmup={lr_warmup_steps} | Init WD(BitLinear)={bitlinear_wd_init}")
    else:
        print(f"LR Constante: {lr:.2e}")
    print("=" * 80)

    cfg = HybridConfig(
        seed=seed,
        device=str(device),
        use_scratchpad=True,
        use_explicit_operands=False,
    )
    if device.type == "cpu":
        torch.set_num_threads(cfg.cpu_threads)

    # Conjuntos de datos pre-registrados
    train_dataset = ProceduralCompositionalDataset(cfg, split="train")
    train_200_dataset = ProceduralCompositionalDataset(cfg, split="train")
    train_200_dataset.samples = random.Random(42).sample(train_dataset.unique_records, 200)

    dev_dataset = ProceduralCompositionalDataset(cfg, split="dev")
    test_comp_dataset = ProceduralCompositionalDataset(cfg, split="test-composicional")

    model = TRACECausalModel(cfg).to(device)

    # Optimizador con grupos desacoplados de decaimiento (BitNet b1.58: 3 grupos)
    init_bitlinear_wd = bitlinear_wd_init if use_bitnet_schedule else 0.0
    param_groups = model.configure_adamw_groups(bitlinear_weight_decay=init_bitlinear_wd)
    initial_lr = lr_peak if use_bitnet_schedule else lr
    optimizer = torch.optim.AdamW(param_groups, lr=initial_lr, betas=(0.9, 0.98), eps=1e-8)

    # Desacoplamiento de warmups
    sched_warmup = warmup_steps if warmup_steps is not None else int(steps * warmup_ratio)
    lr_warmup = lr_warmup_steps if use_bitnet_schedule else sched_warmup

    # Creador de lotes
    def get_batch_iterator():
        while True:
            # Mezclar muestras
            indices = torch.randperm(len(train_dataset.samples)).tolist()
            for i in range(0, len(indices) - batch_size + 1, batch_size):
                batch_samples = [train_dataset.samples[idx] for idx in indices[i : i + batch_size]]
                yield collate_fn(
                    batch_samples,
                    cfg.max_seq_len,
                    cfg.pad_token_id,
                    cfg.bos_token_id,
                    cfg.eos_token_id,
                )

    batch_iter = get_batch_iterator()

    print(f"[*] Inicio del entrenamiento. Muestras Train: {len(train_dataset)} | Dev: {len(dev_dataset)} | Sched Warmup: {sched_warmup}")
    t0_start = time.time()
    best_dev_em = -1.0
    best_dev_score = -1.0

    model.train()
    for step in range(1, steps + 1):
        x_0, plen_list = next(batch_iter)
        x_0 = x_0.to(device)
        t_noise = torch.ones(x_0.shape[0], device=device)

        # Currículo de Scheduled Sampling para anclaje de Etapa 1
        if step <= sched_warmup:
            sched_p = 1.0
        else:
            progress = (step - sched_warmup) / max(1, steps - sched_warmup)
            sched_p = max(0.70, 1.0 - 0.30 * progress)

        # Schedule BitNet b1.58: LR Coseno y Weight Decay de 2 Etapas
        if use_bitnet_schedule:
            if step <= lr_warmup:
                curr_lr = lr_peak * (step / max(1, lr_warmup))
            else:
                decay_progress = (step - lr_warmup) / max(1, steps - lr_warmup)
                curr_lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + math.cos(math.pi * decay_progress))
            for pg in optimizer.param_groups:
                pg["lr"] = curr_lr

            # En la segunda mitad del entrenamiento (Fase 2), bajar decay de shadow weights a 0.0
            if step >= (steps // 2):
                for pg in optimizer.param_groups:
                    if pg.get("name") == "bitlinear_shadow":
                        pg["weight_decay"] = 0.0

        # Telemetría de saturación y cambios de signo (flips)
        record_telemetry = (step % 50 == 0 or step == 1 or step == steps or (steps <= 20 and step % 2 == 0))

        optimizer.zero_grad()
        out = model.forward_two_stage(
            x_0, plen_list, t_noise, scheduled_sampling_p=sched_p, record_telemetry=record_telemetry
        )
        loss = out["loss"]
        loss.backward()

        # Gradient clipping para estabilidad
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 50 == 0 or step == 1 or step == steps or (steps <= 20 and step % 2 == 0):
            u_loss = out.get("loss_units", torch.tensor(0.0)).item()
            t_loss = out.get("loss_tens", torch.tensor(0.0)).item()
            h_loss = out.get("loss_hundreds", torch.tensor(0.0)).item()
            telem_str = ""
            if record_telemetry:
                from trace.layers.bitlinear import BitLinear
                sats = [m.last_stats["act_saturation"] for m in model.modules() if isinstance(m, BitLinear) and "act_saturation" in m.last_stats]
                flips = [m.last_stats["code_flip_rate"] for m in model.modules() if isinstance(m, BitLinear) and "code_flip_rate" in m.last_stats]
                avg_sat = (sum(sats) / len(sats)) * 100.0 if sats else 0.0
                avg_flip = (sum(flips) / len(flips)) * 100.0 if flips else 0.0
                telem_str = f" | ActSat: {avg_sat:.1f}% | Flips: {avg_flip:.2f}%"

            current_lr_val = optimizer.param_groups[0]["lr"]
            print(
                f"Paso {step:04d}/{steps} | Loss: {loss.item():.4f} "
                f"(S1: {out['loss_stage1'].item():.4f}, U: {u_loss:.4f}, D: {t_loss:.4f}, C: {h_loss:.4f}) | "
                f"SchedP: {sched_p:.2f} | LR: {current_lr_val:.2e}{telem_str}"
            )

        if step % eval_interval == 0 or step == steps:
            print(f"\n--- EVALUACIÓN PERIÓDICA (Paso {step}) ---")
            train_200_metrics = evaluate_causal(model, train_200_dataset, device)
            print(
                f"  TRAIN-200: ExactMatch={train_200_metrics['exact_match_pct']:.2f}% | "
                f"SP-EM={train_200_metrics['scratchpad_exact_pct']:.2f}% | "
                f"DigAcc={train_200_metrics['digit_acc_pct']:.2f}% | "
                f"ECE={train_200_metrics.get('ece_pct', 0.0):.2f}% | "
                f"S2-ECE={train_200_metrics.get('stage2_ece_pct', 0.0):.2f}%"
            )

            dev_metrics = evaluate_causal(model, dev_dataset, device)
            print(
                f"  DEV: ExactMatch={dev_metrics['exact_match_pct']:.2f}% "
                f"({dev_metrics['em_count']}/{dev_metrics['total_cases']}) | "
                f"ScratchpadEM={dev_metrics['scratchpad_exact_pct']:.2f}% | "
                f"FinalEM={dev_metrics['final_exact_pct']:.2f}% | "
                f"DigAcc={dev_metrics['digit_acc_pct']:.2f}% | "
                f"ECE={dev_metrics.get('ece_pct', 0.0):.2f}% | "
                f"S2-ECE={dev_metrics.get('stage2_ece_pct', 0.0):.2f}% | "
                f"HiConf(>=0.85)={dev_metrics.get('high_conf_acc_pct', 0.0):.1f}% ({dev_metrics.get('high_conf_count', 0)}) | "
                f"LoConf(<0.85)={dev_metrics.get('low_conf_acc_pct', 0.0):.1f}% ({dev_metrics.get('low_conf_count', 0)})"
            )

            # Puntuación compuesta para desempate en fases tempranas
            dev_score = (
                dev_metrics["exact_match_pct"] * 10000.0
                + dev_metrics["final_exact_pct"] * 1000.0
                + dev_metrics["scratchpad_exact_pct"] * 100.0
                + dev_metrics["digit_acc_pct"]
            )

            if dev_score > best_dev_score:
                best_dev_score = dev_score
                best_dev_em = dev_metrics["exact_match_pct"]
                print(f"  [*] Guardando mejor checkpoint (Score={dev_score:.2f}, SP_EM={dev_metrics['scratchpad_exact_pct']:.2f}%) en: {checkpoint_out}")
                torch.save(
                    {
                        "step": step,
                        "model_state": model.state_dict(),
                        "cfg": cfg,
                        "best_dev_em": best_dev_em,
                        "best_dev_score": best_dev_score,
                        "dev_metrics": dev_metrics,
                    },
                    checkpoint_out,
                )
            model.train()

        # Liberar memoria al final de la iteración de evaluación
        if step % eval_interval == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    total_time = time.time() - t0_start
    print("\n" + "=" * 80)
    print(f"Entrenamiento completado en {total_time:.2f} s ({total_time / steps * 1000:.1f} ms/paso)")
    print(f"Mejor Exact Match en DEV: {best_dev_em:.2f}% (Mejor Score: {best_dev_score:.2f})")
    print("=" * 80)

    # Guardar siempre el último estado al finalizar
    last_ckpt_path = checkpoint_out.replace(".pt", "_last.pt")
    torch.save(
        {
            "step": steps,
            "model_state": model.state_dict(),
            "cfg": cfg,
            "dev_metrics": dev_metrics,
        },
        last_ckpt_path,
    )
    print(f"[*] Último estado guardado en: {last_ckpt_path}")

    # Evaluación final en Test-Composicional
    print("\n--- EVALUACIÓN FINAL EN TEST-COMPOSICIONAL (480 casos) ---")
    if os.path.exists(checkpoint_out):
        ckpt = torch.load(checkpoint_out, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    comp_metrics = evaluate_causal(model, test_comp_dataset, device)
    print(
        f"  TEST-COMP: ExactMatch={comp_metrics['exact_match_pct']:.2f}% "
        f"({comp_metrics['em_count']}/{comp_metrics['total_cases']}) | "
        f"ScratchpadEM={comp_metrics['scratchpad_exact_pct']:.2f}% | "
        f"FinalEM={comp_metrics['final_exact_pct']:.2f}% | "
        f"DigAcc={comp_metrics['digit_acc_pct']:.2f}% | "
        f"ECE={comp_metrics.get('ece_pct', 0.0):.2f}% | "
        f"S2-ECE={comp_metrics.get('stage2_ece_pct', 0.0):.2f}% | "
        f"HiConf(>=0.85)={comp_metrics.get('high_conf_acc_pct', 0.0):.1f}% ({comp_metrics.get('high_conf_count', 0)}) | "
        f"LoConf(<0.85)={comp_metrics.get('low_conf_acc_pct', 0.0):.1f}% ({comp_metrics.get('low_conf_count', 0)})"
    )

    return comp_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento Causal de 2 Etapas — Basic-Lite TRACE")
    parser.add_argument("--steps", type=int, default=1000, help="Pasos de entrenamiento")
    parser.add_argument("--lr", type=float, default=3e-4, help="Tasa de aprendizaje (estándar)")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamaño de lote")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--device", type=str, default="auto", help="cpu o cuda")
    parser.add_argument("--eval_interval", type=int, default=200, help="Intervalo de evaluación")
    parser.add_argument("--warmup_ratio", type=float, default=0.40, help="Fracción de pasos con sched_p = 1.0 (si warmup_steps es None)")
    parser.add_argument("--warmup_steps", type=int, default=None, help="Pasos fijos con sched_p = 1.0 (anclaje perfecto de subtotales)")
    parser.add_argument("--lr_warmup_steps", type=int, default=300, help="Pasos de rampa lineal para LR hacia lr_peak")
    parser.add_argument("--bitnet_schedule", action="store_true", help="Activar schedule QAT de 2 etapas estilo BitNet b1.58")
    parser.add_argument("--lr_peak", type=float, default=1.0e-3, help="LR pico para schedule BitNet")
    parser.add_argument("--lr_min", type=float, default=1.5e-4, help="LR mínimo para decaimiento coseno")
    parser.add_argument("--bitlinear_wd", type=float, default=0.05, help="Weight decay inicial para shadow weights de BitLinear")
    parser.add_argument("--out", type=str, default="trace_checkpoint.pt", help="Ruta de checkpoint")
    args = parser.parse_args()

    train(
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        device_name=args.device,
        eval_interval=args.eval_interval,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        lr_warmup_steps=args.lr_warmup_steps,
        use_bitnet_schedule=args.bitnet_schedule,
        lr_peak=args.lr_peak,
        lr_min=args.lr_min,
        bitlinear_wd_init=args.bitlinear_wd,
        checkpoint_out=args.out,
    )
