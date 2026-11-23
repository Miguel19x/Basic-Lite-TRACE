#!/usr/bin/env python3
# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Pipeline Universal de Entrenamiento Multi-Hardware para TRACELanguageModel (Hito 3.6).
Soporta CUDA (T4/L4/A100), ROCm, Intel XPU, Apple MPS y CPU, con DDP nativo (torchrun),
Continual QAT (16-bit -> 1.58-bit), acumulación de gradientes, AdamW fusionado y
exportación directa a Paged BitCOS v1.1.
"""

import argparse
import contextlib
import dataclasses
import inspect
import json
import math
import os
import time
from trace.lm.config import TRACELMConfig
from trace.lm.model import TRACELanguageModel

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from export_paged_lm import export_lm_checkpoint


class MemmapDataset(Dataset):
    """Dataset seguro para multiproceso sobre tokens uint16 en bloques fijos [seq_len]."""

    def __init__(self, data_path: str, seq_len: int):
        self.data_path = data_path
        self.seq_len = seq_len
        file_bytes = os.path.getsize(data_path)
        self.total_tokens = file_bytes // np.dtype(np.uint16).itemsize
        self.n_samples = max(0, (self.total_tokens - 1) // seq_len)
        self._mmap: np.memmap | None = None

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._mmap is None:
            self._mmap = np.memmap(self.data_path, dtype=np.uint16, mode="r")
        start = idx * self.seq_len
        chunk = torch.from_numpy(self._mmap[start : start + self.seq_len + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]


def unwrap_model(model: torch.nn.Module) -> TRACELanguageModel:
    """Desenvuelve DDP (.module) y torch.compile (._orig_mod) de forma segura."""
    while hasattr(model, "module") or hasattr(model, "_orig_mod"):
        if hasattr(model, "module"):
            model = model.module
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
    return model  # type: ignore[return-value]


def setup_distributed() -> tuple[bool, int, int, int]:
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return True, int(os.environ.get("RANK", 0)), int(os.environ["WORLD_SIZE"]), local_rank
    return False, 0, 1, 0


def get_device(local_rank: int = 0) -> tuple[torch.device, str]:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}"), "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device(f"xpu:{local_rank}"), "xpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def get_autocast_context(device_type: str, amp_dtype: torch.dtype, use_amp: bool):
    if not use_amp:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=True)


def configure_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    device_type: str,
) -> torch.optim.Optimizer:
    raw_model = unwrap_model(model)
    param_groups = raw_model.configure_adamw_groups(
        bitlinear_weight_decay=weight_decay,
        weight_decay=weight_decay,
    )
    adamw_params = inspect.signature(torch.optim.AdamW.__init__).parameters
    use_fused = device_type == "cuda" and "fused" in adamw_params
    kwargs: dict = {"lr": lr, "betas": (0.9, 0.95), "eps": 1e-8}
    if use_fused:
        kwargs["fused"] = True
    return torch.optim.AdamW(param_groups, **kwargs)


def set_bitlinear_weight_decay(optimizer: torch.optim.Optimizer, new_wd: float) -> None:
    for group in optimizer.param_groups:
        if group.get("name") == "bitlinear_shadow":
            group["weight_decay"] = new_wd


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr: float,
    base_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        denom = max(1, max_steps - warmup_steps)
        progress = min(1.0, max(0.0, float(current_step - warmup_steps) / float(denom)))
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (min_lr + coeff * (base_lr - min_lr)) / max(base_lr, 1e-12)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler | None,
    cfg: TRACELMConfig,
    step: int,
    best_val_loss: float,
) -> None:
    raw_model = unwrap_model(model)
    payload = {
        "step": step,
        "model_state": raw_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "cfg": dataclasses.asdict(cfg),
        "best_val_loss": best_val_loss,
    }
    torch.save(payload, path)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    device_type: str,
    amp_dtype: torch.dtype,
    use_amp: bool,
    is_dist: bool,
    max_batches: int = 20,
) -> float:
    model.eval()
    loss_sum = torch.zeros(1, device=device, dtype=torch.float32)
    count = torch.zeros(1, device=device, dtype=torch.float32)

    for idx, (x, y) in enumerate(loader):
        if idx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with get_autocast_context(device_type, amp_dtype, use_amp):
            _, loss, _ = model(x, targets=y)
        loss_sum += loss.detach().float()
        count += 1.0

    if is_dist:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)

    model.train()
    n = count.item()
    return (loss_sum.item() / n) if n > 0 else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento Multi-Hardware TRACELanguageModel")
    parser.add_argument("--train-data", type=str, default="data/train.bin", help="Ruta a train.bin")
    parser.add_argument("--val-data", type=str, default="data/val.bin", help="Ruta a val.bin")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directorio de checkpoints")
    parser.add_argument("--log-file", type=str, default="results/train_lm.jsonl", help="Archivo de logs JSONL")

    # Arquitectura y QAT
    parser.add_argument("--preset", type=str, default="25m", choices=["dev", "25m", "25m-bi", "45m-bi"], help="Preset TRACELMConfig")
    parser.add_argument("--weight-scale", type=str, default="mean", choices=["mean", "median"], help="Escala W1.58")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Largo máximo de contexto")
    parser.add_argument("--fp16-warmup-ratio", type=float, default=0.0, help="Fracción Continual QAT 16b->1.58b")
    parser.add_argument("--wd-stage2-ratio", type=float, default=0.0, help="Fracción para apagar WD BitLinear (0=off)")

    # Hiperparámetros calibrados para BitNet <100M
    parser.add_argument("--batch-size", type=int, default=16, help="Micro-batch por dispositivo")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Pasos de acumulación de gradiente")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate pico (base BitNet <100M)")
    parser.add_argument("--min-lr", type=float, default=1e-4, help="Learning rate mínimo coseno")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay AdamW")
    parser.add_argument("--max-steps", type=int, default=None, help="Pasos de optimización totales")
    parser.add_argument("--warmup-steps", type=int, default=200, help="Pasos de warmup lineal de LR")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Clip de norma global de gradientes")
    parser.add_argument("--log-every", type=int, default=20, help="Frecuencia de logging (pasos)")
    parser.add_argument("--eval-every", type=int, default=200, help="Frecuencia de validación (pasos)")
    parser.add_argument("--save-every", type=int, default=500, help="Frecuencia de guardado (pasos)")

    # Flags operativos
    parser.add_argument("--smoke", action="store_true", help="Smoke test rápido en CPU")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint .pt para reanudar")
    parser.add_argument("--compile", action="store_true", help="Activar torch.compile()")
    parser.add_argument("--grad-checkpoint", action="store_true", help="Activar gradient checkpointing")
    parser.add_argument("--no-amp", action="store_true", help="Desactivar mixed precision")
    parser.add_argument("--no-export", action="store_true", help="Omitir exportación final a Paged BitCOS")

    args = parser.parse_args()

    is_dist, rank, world_size, local_rank = setup_distributed()
    device, device_type = get_device(local_rank)
    if device_type == "cpu":
        torch.set_num_threads(2)

    if args.max_steps is None:
        args.max_steps = 5 if args.smoke else 5000

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.log_file)), exist_ok=True)

    # 1. Configuración (o recuperación desde checkpoint si se reanuda)
    ckpt: dict | None = None
    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Checkpoint de reanudación no encontrado: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)

    if ckpt is not None and "cfg" in ckpt:
        raw_cfg = ckpt["cfg"]
        cfg = TRACELMConfig(**raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg
    else:
        if args.preset == "25m":
            cfg = TRACELMConfig.preset_25m()
        elif args.preset == "25m-bi":
            cfg = TRACELMConfig.preset_25m_bilingual()
        elif args.preset == "45m-bi":
            cfg = TRACELMConfig.preset_45m_bilingual()
        else:
            cfg = TRACELMConfig()
        if args.max_seq_len is not None:
            cfg.max_seq_len = args.max_seq_len
        cfg.weight_scale_type = args.weight_scale

    if args.smoke:
        if ckpt is None:
            cfg.d_model = 64
            cfg.n_layers = 2
            cfg.n_heads = 2
            cfg.d_ff = 128
            cfg.kv_lora_rank = 32
            cfg.qk_nope_head_dim = 16
            cfg.qk_rope_head_dim = 8
            cfg.v_head_dim = 16
            cfg.max_seq_len = 64
        args.batch_size = 2
        args.warmup_steps = min(args.warmup_steps, 2)
        args.log_every = 1
        args.eval_every = args.max_steps
        args.save_every = args.max_steps

    use_amp = (not args.no_amp) and device_type in ("cuda", "xpu")
    if device_type == "cuda":
        is_bf16_native = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] >= 8
            and torch.cuda.is_bf16_supported()
        )
        amp_dtype = torch.bfloat16 if is_bf16_native else torch.float16
    elif device_type == "xpu":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float32

    continual_warmup_steps = int(args.max_steps * args.fp16_warmup_ratio)
    wd_stage2_step = int(args.max_steps * args.wd_stage2_ratio) if args.wd_stage2_ratio > 0.0 else 0

    if rank == 0:
        print("=" * 80)
        print("Basic-Lite TRACE — Entrenamiento Multi-Hardware (Hito 3.6)")
        print(f"Dispositivo:   {device} ({device_type.upper()}) | AMP: {amp_dtype if use_amp else 'FP32'} | DDP: {is_dist} (x{world_size})")
        print(f"Modelo:        L={cfg.n_layers}, D={cfg.d_model}, d_c={cfg.kv_lora_rank}, seq={cfg.max_seq_len}, scale={cfg.weight_scale_type}")
        print(f"Optimización:  LR={args.lr}, WD={args.weight_decay}, steps={args.max_steps}, grad_accum={args.grad_accum_steps}")
        print(f"Continual QAT: warmup={continual_warmup_steps} pasos | WD-Stage2={wd_stage2_step} | GradCkpt={args.grad_checkpoint}")
        print("=" * 80)

    # 2. Instanciación y restauración de estado
    model = TRACELanguageModel(cfg)
    start_step = 0
    best_val_loss = float("inf")

    if ckpt is not None:
        state_key = "model_state" if "model_state" in ckpt else "model_state_dict"
        model.load_state_dict(ckpt[state_key], strict=True)
        start_step = int(ckpt.get("step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        if rank == 0:
            print(f"[Resume] Restaurado desde {args.resume} (paso {start_step}, best_val_loss={best_val_loss:.4f})")

    if continual_warmup_steps > 0 and start_step < continual_warmup_steps:
        model.set_quantize_weights(False)
        if rank == 0:
            print(f"[Continual QAT] Modo continuo activo hasta el paso {continual_warmup_steps}.")
    else:
        model.set_quantize_weights(True)

    model.to(device)

    if is_dist:
        model = DDP(model, device_ids=[local_rank] if device_type == "cuda" else None)

    if args.compile and hasattr(torch, "compile"):
        if rank == 0:
            print("[Compile] Compilando grafo con torch.compile()...")
        model = torch.compile(model)

    # 3. Optimizador, Scheduler y GradScaler
    optimizer = configure_optimizer(model, args.lr, args.weight_decay, device_type)
    scheduler = get_lr_scheduler(optimizer, args.warmup_steps, args.max_steps, args.min_lr, args.lr)
    use_scaler = use_amp and device_type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_scaler else None

    if ckpt is not None:
        opt_key = "optimizer_state" if "optimizer_state" in ckpt else "optimizer_state_dict"
        if opt_key in ckpt and ckpt[opt_key] is not None:
            optimizer.load_state_dict(ckpt[opt_key])
        if "scheduler_state" in ckpt and ckpt["scheduler_state"] is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if scaler is not None and ckpt.get("scaler_state") is not None:
            scaler.load_state_dict(ckpt["scaler_state"])

    if wd_stage2_step > 0 and start_step >= wd_stage2_step:
        set_bitlinear_weight_decay(optimizer, 0.0)

    # 4. DataLoaders sobre np.memmap
    train_ds = MemmapDataset(args.train_data, cfg.max_seq_len)
    val_ds = MemmapDataset(args.val_data, cfg.max_seq_len)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(f"Dataset insuficiente para seq_len={cfg.max_seq_len}: train={len(train_ds)}, val={len(val_ds)}")

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_dist else None
    num_workers = 2 if (device_type == "cuda" and not args.smoke) else 0
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=(device_type == "cuda"),
        persistent_workers=(num_workers > 0),
        drop_last=(len(train_ds) >= args.batch_size),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device_type == "cuda"),
    )

    # 5. Bucle de Entrenamiento (cero .item() por iteración)
    model.train()
    step = start_step
    epoch = 0
    running_loss = torch.zeros(1, device=device, dtype=torch.float32)
    steps_since_log = 0
    accum_tokens = 0
    t0 = time.perf_counter()
    train_iter = iter(train_loader)

    def _next_batch() -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal train_iter, epoch
        try:
            return next(train_iter)
        except StopIteration:
            epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_iter = iter(train_loader)
            return next(train_iter)

    while step < args.max_steps:
        if continual_warmup_steps > 0 and step == continual_warmup_steps:
            unwrap_model(model).set_quantize_weights(True)
            if rank == 0:
                print(f"[Continual QAT Paso {step}] Transición completada: cuantización ternaria W1.58 ACTIVA.")

        if wd_stage2_step > 0 and step == wd_stage2_step:
            set_bitlinear_weight_decay(optimizer, 0.0)
            if rank == 0:
                print(f"[WD Stage 2 Paso {step}] Weight decay en BitLinear ajustado a 0.0.")

        optimizer.zero_grad(set_to_none=True)
        step_loss = torch.zeros(1, device=device, dtype=torch.float32)

        for micro in range(args.grad_accum_steps):
            x, y = _next_batch()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            accum_tokens += x.numel() * world_size

            sync_ctx = (
                model.no_sync()
                if (is_dist and isinstance(model, DDP) and micro < args.grad_accum_steps - 1)
                else contextlib.nullcontext()
            )
            with sync_ctx:
                with get_autocast_context(device_type, amp_dtype, use_amp):
                    _, loss, _ = model(x, targets=y, use_grad_checkpoint=args.grad_checkpoint)
                scaled_loss = loss / float(args.grad_accum_steps)
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
            step_loss += loss.detach().float() / float(args.grad_accum_steps)

        if scaler is not None:
            if args.grad_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        scheduler.step()
        running_loss += step_loss
        steps_since_log += 1
        step += 1

        # Logging sincronizado solo cada log_every pasos
        if step % args.log_every == 0 or step == args.max_steps:
            if is_dist:
                dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
                avg_loss = (running_loss.item() / float(world_size)) / float(steps_since_log)
            else:
                avg_loss = running_loss.item() / float(steps_since_log)

            dt = max(1e-6, time.perf_counter() - t0)
            tok_per_sec = float(accum_tokens) / dt
            curr_lr = scheduler.get_last_lr()[0]

            if rank == 0:
                q_tag = "W1.58" if unwrap_model(model).cfg.quantize_weights_enabled else "FP16W"
                print(
                    f"Paso {step:5d}/{args.max_steps} [{q_tag}] | "
                    f"Loss: {avg_loss:.4f} | LR: {curr_lr:.2e} | {tok_per_sec:.1f} tok/s"
                )
                with open(args.log_file, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps({
                            "step": step,
                            "mode": q_tag,
                            "train_loss": avg_loss,
                            "lr": curr_lr,
                            "tok_per_sec": tok_per_sec,
                            "timestamp": time.time(),
                        }) + "\n"
                    )

            running_loss.zero_()
            steps_since_log = 0
            accum_tokens = 0
            t0 = time.perf_counter()

        # Validación periódica
        if step % args.eval_every == 0 or step == args.max_steps:
            val_loss = evaluate(model, val_loader, device, device_type, amp_dtype, use_amp, is_dist)
            val_ppl = math.exp(min(val_loss, 20.0))
            if rank == 0:
                print(f"--- [Val Paso {step}] Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} (Mejor: {best_val_loss:.4f}) ---")
                with open(args.log_file, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps({
                            "step": step,
                            "val_loss": val_loss,
                            "val_ppl": val_ppl,
                            "timestamp": time.time(),
                        }) + "\n"
                    )
                can_save_best = (step >= continual_warmup_steps) or (step == args.max_steps)
                if can_save_best and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = os.path.join(args.save_dir, "best_model.pt")
                    save_checkpoint(best_path, model, optimizer, scheduler, scaler, cfg, step, best_val_loss)
                    print(f"✓ Mejor checkpoint actualizado en {best_path}")

        # Checkpoint periódico y puntero fijo last.pt
        if (step % args.save_every == 0 or step == args.max_steps) and rank == 0:
            ckpt_path = os.path.join(args.save_dir, f"checkpoint_step_{step}.pt")
            last_path = os.path.join(args.save_dir, "last.pt")
            save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, cfg, step, best_val_loss)
            save_checkpoint(last_path, model, optimizer, scheduler, scaler, cfg, step, best_val_loss)
            print(f"✓ Checkpoint guardado en {ckpt_path} (y {last_path})")

    # 6. Exportación final a Paged BitCOS v1.1
    if rank == 0 and not args.no_export:
        unwrap_model(model).set_quantize_weights(True)
        best_path = os.path.join(args.save_dir, "best_model.pt")
        final_ckpt = best_path if os.path.exists(best_path) else os.path.join(args.save_dir, "last.pt")
        output_bin = os.path.join(args.save_dir, "trace_lm_trained_paged_bitcos.bin")
        print("\n" + "=" * 80)
        print(f"Exportando {final_ckpt} -> {output_bin}...")
        export_lm_checkpoint(checkpoint_path=final_ckpt, output_path=output_bin)
        print("=" * 80)

    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
