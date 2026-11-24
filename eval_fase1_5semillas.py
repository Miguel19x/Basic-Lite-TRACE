#!/usr/bin/env python3
# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available

"""
Evaluación exploratoria multi-semilla de Fase 1 (Basic-Lite TRACE).
Evalúa las 5 semillas bajo el schedule BitNet b1.58 de 2 etapas (paso 6000 o best dev) en:
  1. Train-200 (muestreado sin reemplazo, semilla 42)
  2. DEV (108 muestras)
  3. Test-Azar (240 muestras)
  4. Test-Composicional (480 muestras)
Calcula métricas por semilla, calibración ECE desacoplada, desglose por los 8 pares no vistos,
agregados estadísticos con ddof=1, IC95 Bootstrap (10,000 remuestreos deterministas) y
exporta atómicamente el artefacto canónico results/eval_fase1_bitnet_5semillas_results.json.
"""

import argparse
import json
import os
import random
import time
from typing import Any

import numpy as np
import torch

from config import HybridConfig
from data import ProceduralCompositionalDataset, parse_fixed_scratchpad
from eval import compute_ece, evaluate_causal
from trace.models.causal import TRACECausalModel

# Concurrencia obligatoria en CPU i3-1115G4 (§2.3)
torch.set_num_threads(2)


def compute_bootstrap_ci95(values: list[float], n_bootstrap: int = 10000, seed: int = 42) -> tuple[float, float]:
    if len(values) <= 1:
        v = values[0] if values else 0.0
        return float(v), float(v)
    rng = np.random.RandomState(seed)
    vals = np.array(values, dtype=np.float64)
    n = len(vals)
    boot_means = [float(np.mean(rng.choice(vals, size=n, replace=True))) for _ in range(n_bootstrap)]
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return round(ci_low, 2), round(ci_high, 2)


def compute_stats(vals: list[float]) -> dict[str, Any]:
    n = len(vals)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "ci95_parametric": [0.0, 0.0], "ci95_bootstrap": [0.0, 0.0], "values": []}
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    median = float(np.median(vals))
    v_min = float(np.min(vals))
    v_max = float(np.max(vals))
    t_crit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 1.96)
    margin = float(t_crit * (std / np.sqrt(n))) if n > 1 else 0.0
    return {
        "mean": round(mean, 2),
        "std": round(std, 2),
        "median": round(median, 2),
        "min": round(v_min, 2),
        "max": round(v_max, 2),
        "ci95_parametric": [round(mean - margin, 2), round(mean + margin, 2)],
        "ci95_bootstrap": list(compute_bootstrap_ci95(vals)),
        "values": [round(float(x), 2) for x in vals],
    }


def evaluate_test_comp_detailed(
    model: TRACECausalModel,
    dataset: ProceduralCompositionalDataset,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """
    Evaluación en una sola pasada de Test-Composicional.
    Calcula simultáneamente métricas base, calibración desacoplada y desglose por par no visto.
    """
    model.eval()
    total_cases = len(dataset)
    em_total_count = 0
    sp_em_count = 0
    final_em_count = 0
    fmt_valid_count = 0

    total_digits = 0
    matching_digits = 0
    total_time_ms = 0.0

    all_confidences: list[float] = []
    all_correctness: list[float] = []
    s2_confidences: list[float] = []
    s2_correctness: list[float] = []

    pair_perf = {f"{p}_x_{c}": {"ok": 0, "total": 0, "pct": 0.0} for p, c in ProceduralCompositionalDataset.HELD_OUT_PAIRS}

    digit_indices = [
        (4, 7),    # pw (dígitos 0, 1, 2)
        (13, 16),  # pms (dígitos 3, 4, 5)
        (23, 26),  # watt (dígitos 6, 7, 8 - Etapa 2)
        (31, 34),  # ms (dígitos 9, 10, 11 - Etapa 2)
    ]

    samples = dataset.samples

    with torch.no_grad():
        for idx in range(total_cases):
            prompt_b, target_b, expected_watt, expected_ms = samples[idx]
            sub_w, sub_ms, _, _ = dataset.record_meta.get(
                (prompt_b, target_b), (None, None, expected_watt, expected_ms)
            )

            p_str = prompt_b.decode("utf-8")
            cur_p, cur_c = "", ""
            for line in p_str.splitlines():
                if line.startswith("perfil:"):
                    cur_p = line.split(":", 1)[1].strip()
                elif line.startswith("carga:"):
                    cur_c = line.split(":", 1)[1].strip()
            pair_key = f"{cur_p}_x_{cur_c}"
            if pair_key in pair_perf:
                pair_perf[pair_key]["total"] += 1

            t0 = time.perf_counter()
            gen_str, meta = model.generate_causal_two_step(prompt_b)
            t1 = time.perf_counter()
            total_time_ms += (t1 - t0) * 1000.0

            valid, pred_pw, pred_pms, pred_watt, pred_ms = parse_fixed_scratchpad(gen_str)

            sp_match = False
            fn_match = False

            if valid:
                fmt_valid_count += 1
                sp_match = (pred_pw == sub_w) and (pred_pms == sub_ms)
                if sp_match:
                    sp_em_count += 1

                fn_match = (pred_watt == expected_watt) and (pred_ms == expected_ms)
                if fn_match:
                    final_em_count += 1

                if sp_match and fn_match:
                    em_total_count += 1
                    if pair_key in pair_perf:
                        pair_perf[pair_key]["ok"] += 1

            expected_target_str = target_b.decode("utf-8")
            case_correctness: list[float] = []
            for s_start, s_end in digit_indices:
                exp_sub = expected_target_str[s_start:s_end]
                pred_sub = gen_str[s_start:s_end] if len(gen_str) >= s_end else "???"
                for e_char, p_char in zip(exp_sub, pred_sub):
                    total_digits += 1
                    is_match = 1.0 if e_char == p_char else 0.0
                    if e_char == p_char:
                        matching_digits += 1
                    case_correctness.append(is_match)

            if "digit_confidences" in meta and len(meta["digit_confidences"]) == 12 and len(case_correctness) == 12:
                all_confidences.extend(meta["digit_confidences"])
                all_correctness.extend(case_correctness)
                s2_confidences.extend(meta["digit_confidences"][6:12])
                s2_correctness.extend(case_correctness[6:12])

    em_pct = (em_total_count / total_cases) * 100.0
    sp_em_pct = (sp_em_count / total_cases) * 100.0
    final_em_pct = (final_em_count / total_cases) * 100.0
    fmt_valid_pct = (fmt_valid_count / total_cases) * 100.0
    digit_acc_pct = (matching_digits / total_digits) * 100.0 if total_digits > 0 else 0.0
    ms_per_case = total_time_ms / total_cases if total_cases > 0 else 0.0

    ece_pct = compute_ece(all_confidences, all_correctness)
    stage2_ece_pct = compute_ece(s2_confidences, s2_correctness)

    high_conf_correct, high_conf_total = 0, 0
    low_conf_correct, low_conf_total = 0, 0
    for conf, corr in zip(all_confidences, all_correctness):
        if conf >= 0.85:
            high_conf_total += 1
            if corr == 1.0:
                high_conf_correct += 1
        else:
            low_conf_total += 1
            if corr == 1.0:
                low_conf_correct += 1

    high_conf_acc = (high_conf_correct / high_conf_total * 100.0) if high_conf_total > 0 else 0.0
    low_conf_acc = (low_conf_correct / low_conf_total * 100.0) if low_conf_total > 0 else 0.0

    for k, v in pair_perf.items():
        v["pct"] = round((v["ok"] / v["total"] * 100.0) if v["total"] > 0 else 0.0, 2)

    base_metrics = {
        "exact_match_pct": em_pct,
        "scratchpad_exact_pct": sp_em_pct,
        "final_exact_pct": final_em_pct,
        "fmt_valid_pct": fmt_valid_pct,
        "digit_acc_pct": digit_acc_pct,
        "ms_per_case": ms_per_case,
        "em_count": em_total_count,
        "total_cases": total_cases,
        "ece_pct": ece_pct,
        "stage2_ece_pct": stage2_ece_pct,
        "high_conf_acc_pct": high_conf_acc,
        "high_conf_count": high_conf_total,
        "low_conf_acc_pct": low_conf_acc,
        "low_conf_count": low_conf_total,
    }
    return base_metrics, pair_perf


def main():
    parser = argparse.ArgumentParser(description="Evaluación 5 semillas Fase 1 BitNet (Basic-Lite TRACE)")
    parser.add_argument("--variant", choices=["best", "last"], default="last", help="Variante de checkpoints a evaluar (default: last)")
    args = parser.parse_args()

    device = torch.device("cpu")
    seeds = [42, 123, 456, 789, 999]
    suffix = "_last" if args.variant == "last" else ""
    ckpt_paths = {s: f"checkpoints/trace_s{s}_fase1_bitnet{suffix}.pt" for s in seeds}
    out_file = "results/eval_fase1_bitnet_5semillas_results.json" if args.variant == "last" else "results/eval_fase1_5semillas_results.json"

    print("=" * 85)
    print(f"Basic-Lite TRACE — Consolidación 5 Semillas (Variante: {args.variant})")
    print(f"Dispositivo: CPU (2 hilos i3-1115G4) | Semillas: {seeds}")
    print("=" * 85)

    base_cfg = HybridConfig(cpu_threads=2, device="cpu", use_scratchpad=True, use_explicit_operands=False)

    print("[*] Cargando particiones de datos pre-registradas...")
    train_200 = ProceduralCompositionalDataset(base_cfg, split="train")
    train_200.samples = random.Random(42).sample(train_200.unique_records, 200)

    dev_ds = ProceduralCompositionalDataset(base_cfg, split="dev")
    test_azar_ds = ProceduralCompositionalDataset(base_cfg, split="test-azar")
    test_comp_ds = ProceduralCompositionalDataset(base_cfg, split="test-composicional")

    per_seed_results = {}
    arch_cfg_saved = None
    print(f"[*] Datasets listos: Train-200={len(train_200)}, DEV={len(dev_ds)}, Test-Azar={len(test_azar_ds)}, Test-Comp={len(test_comp_ds)}\n")

    for s in seeds:
        path = ckpt_paths[s]
        assert os.path.exists(path), f"Checkpoint no encontrado: {path}"
        print(f"---> Evaluando Semilla {s} ({args.variant}) desde: {path}")
        t0 = time.time()
        if hasattr(torch.serialization, "safe_globals"):
            with torch.serialization.safe_globals([HybridConfig]):
                ckpt = torch.load(path, map_location=device, weights_only=True)
        else:
            ckpt = torch.load(path, map_location=device, weights_only=True)

        raw_cfg = ckpt.get("cfg")
        if raw_cfg is not None:
            ckpt_cfg = HybridConfig(**raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg
            ckpt_cfg.device = "cpu"
            ckpt_cfg.cpu_threads = 2
            model = TRACECausalModel(ckpt_cfg).to(device)
            if arch_cfg_saved is None:
                arch_cfg_saved = vars(ckpt_cfg) if hasattr(ckpt_cfg, "__dict__") else str(ckpt_cfg)
        else:
            model = TRACECausalModel(base_cfg).to(device)
            
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        ckpt_step = ckpt.get("step")

        train_200_m = evaluate_causal(model, train_200, device)
        dev_m = evaluate_causal(model, dev_ds, device)
        test_azar_m = evaluate_causal(model, test_azar_ds, device)
        test_comp_m, pair_perf = evaluate_test_comp_detailed(model, test_comp_ds, device)
        dt = time.time() - t0

        # Sanity check: paso y dev_metrics
        sanity_step_ok = (ckpt_step == 6000) if args.variant == "last" else True
        if args.variant == "last" and not sanity_step_ok:
            print(f"     [!] ADVERTENCIA SANITY CHECK: ckpt step={ckpt_step} != 6000")

        ckpt_dev_em = None
        if "dev_metrics" in ckpt and isinstance(ckpt["dev_metrics"], dict):
            ckpt_dev_em = ckpt["dev_metrics"].get("exact_match_pct")
        elif ckpt.get("best_dev_em") is not None:
            ckpt_dev_em = ckpt.get("best_dev_em")

        sanity_dev_match = False
        if ckpt_dev_em is not None:
            dev_diff = abs(dev_m["exact_match_pct"] - ckpt_dev_em)
            sanity_dev_match = (dev_diff < 1e-4)
            if not sanity_dev_match:
                print(f"     [!] ADVERTENCIA SANITY CHECK: DEV ({dev_m['exact_match_pct']:.2f}%) != ckpt dev_metrics ({ckpt_dev_em:.2f}%) (Δ={dev_diff:.4f}%)")
            else:
                print(f"     [✓] Sanity check DEV OK: coincide exactamente con ckpt dev_metrics ({ckpt_dev_em:.2f}%)")

        per_seed_results[str(s)] = {
            "checkpoint": path,
            "step": ckpt_step,
            "sanity_step_6000": sanity_step_ok,
            "ckpt_dev_em": ckpt_dev_em,
            "sanity_dev_match": sanity_dev_match,
            "eval_time_sec": round(dt, 2),
            "train_200": train_200_m,
            "dev": dev_m,
            "test_azar": test_azar_m,
            "test_comp": test_comp_m,
            "held_out_pair_performance": pair_perf,
        }

        print(f"     [s{s}] Train-200 EM: {train_200_m['exact_match_pct']:.2f}% | DEV EM: {dev_m['exact_match_pct']:.2f}% | Test-Azar EM: {test_azar_m['exact_match_pct']:.2f}% | Test-Comp EM: {test_comp_m['exact_match_pct']:.2f}% ({test_comp_m['em_count']}/480) [ECE={test_comp_m['ece_pct']:.2f}%] ({dt:.1f}s)")

    # Agregados estadísticos
    train_200_em = [r["train_200"]["exact_match_pct"] for r in per_seed_results.values()]
    dev_em = [r["dev"]["exact_match_pct"] for r in per_seed_results.values()]
    test_azar_em = [r["test_azar"]["exact_match_pct"] for r in per_seed_results.values()]
    test_comp_em = [r["test_comp"]["exact_match_pct"] for r in per_seed_results.values()]
    test_comp_sp = [r["test_comp"]["scratchpad_exact_pct"] for r in per_seed_results.values()]
    test_comp_dig = [r["test_comp"]["digit_acc_pct"] for r in per_seed_results.values()]
    test_comp_ece = [r["test_comp"]["ece_pct"] for r in per_seed_results.values()]
    test_comp_s2_ece = [r["test_comp"]["stage2_ece_pct"] for r in per_seed_results.values()]

    aggregate = {
        "train_200_em": compute_stats(train_200_em),
        "dev_em": compute_stats(dev_em),
        "test_azar_em": compute_stats(test_azar_em),
        "test_comp_em": compute_stats(test_comp_em),
        "test_comp_sp_em": compute_stats(test_comp_sp),
        "test_comp_digit_acc": compute_stats(test_comp_dig),
        "test_comp_ece": compute_stats(test_comp_ece),
        "test_comp_stage2_ece": compute_stats(test_comp_s2_ece),
    }

    gate_t200_passed = bool(aggregate["train_200_em"]["mean"] > 90.0)
    gate_tcomp_passed = bool(aggregate["test_comp_em"]["mean"] > 60.0)
    gate_ece_registered = bool(len(test_comp_ece) == len(seeds))
    all_gates_passed = gate_t200_passed and gate_tcomp_passed and gate_ece_registered

    verdict_str = "APROBADO (Train-200 > 90%, Test-Comp > 60%, Calibración ECE Registrada)" if all_gates_passed else "NO ALCANZADO"
    gain = round(aggregate["test_comp_em"]["mean"] - 19.04, 2)

    out_payload = {
        "experiment": f"TRACE_Fase1_BitNet_5Semillas_{args.variant.capitalize()}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": "cpu (Intel Core i3-1115G4, 2 hilos)",
        "classification": "EXPLORATORIO (Fase 1 BitNet 2-etapas, post-prereg-v1, §9.4)",
        "variant": args.variant,
        "verdict_exit_gate": verdict_str,
        "gates_status": {
            "train_200_em_gt_90": {
                "required": "> 90.0%",
                "observed": aggregate["train_200_em"]["mean"],
                "passed": gate_t200_passed,
            },
            "test_comp_em_gt_60": {
                "required": "> 60.0%",
                "observed": aggregate["test_comp_em"]["mean"],
                "passed": gate_tcomp_passed,
            },
            "ece_registered": {
                "required": "Registrada formalmente",
                "observed_mean_ece": aggregate["test_comp_ece"]["mean"],
                "observed_mean_s2_ece": aggregate["test_comp_stage2_ece"]["mean"],
                "passed": gate_ece_registered,
            },
            "all_gates_passed": all_gates_passed,
        },
        "training_protocol": {
            "steps_per_seed": 6000,
            "batch_size": 32,
            "lr_peak": 1e-3,
            "lr_min": 1.5e-4,
            "lr_warmup_steps": 300,
            "warmup_steps": 1200,
            "bitlinear_wd_init": 0.05,
            "bitlinear_wd_second_half": 0.0,
            "optimizer": "AdamW (betas=(0.9, 0.98), eps=1e-8, 3 parameter groups)",
            "architecture_cfg": arch_cfg_saved,
        },
        "seeds_evaluated": [int(s) for s in per_seed_results.keys()],
        "per_seed_results": per_seed_results,
        "aggregate_summary": aggregate,
        "baseline_comparison_fase2b": {
            "fase2b_test_comp_em_mean": 19.04,
            "fase2b_test_comp_em_ci95_boot": [4.04, 43.12],
            "fase1_test_comp_em_mean": aggregate["test_comp_em"]["mean"],
            "fase1_test_comp_em_ci95_boot": aggregate["test_comp_em"]["ci95_bootstrap"],
            "gain_percentage_points": gain,
        },
    }

    tmp_file = f"{out_file}.tmp"
    os.makedirs("results", exist_ok=True)
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_file, out_file)

    print("\n" + "=" * 85)
    print(f"INFORME AGREGADO CONSOLIDADO — FASE 1 ({args.variant.upper()}) (5 SEMILLAS)")
    print("=" * 85)
    print(f"{'Métrica':<32} | {'Fase 2b (Base)':<18} | {'Fase 1 (BitNet QAT)':<25} | {'Δ Ganancia':<10}")
    print("-" * 85)
    print(f"{'Test-Comp Full-EM':<32} | {'19.04% [4.0, 43.1]':<18} | {aggregate['test_comp_em']['mean']:>5.2f}% ± {aggregate['test_comp_em']['std']:<5.2f}% [{aggregate['test_comp_em']['ci95_bootstrap'][0]}, {aggregate['test_comp_em']['ci95_bootstrap'][1]}] | {gain:>+6.2f}%")
    print(f"{'Test-Comp Scratchpad-EM':<32} | {'98.21% [97.4, 99.0]':<18} | {aggregate['test_comp_sp_em']['mean']:>5.2f}% ± {aggregate['test_comp_sp_em']['std']:<5.2f}% [{aggregate['test_comp_sp_em']['ci95_bootstrap'][0]}, {aggregate['test_comp_sp_em']['ci95_bootstrap'][1]}] |")
    print(f"{'Test-Comp Exactitud Dígitos':<32} | {'88.20% ± 5.61%':<18} | {aggregate['test_comp_digit_acc']['mean']:>5.2f}% ± {aggregate['test_comp_digit_acc']['std']:<5.2f}% |")
    print(f"{'Train-200 Full-EM':<32} | {'24.90% ± 26.23%':<18} | {aggregate['train_200_em']['mean']:>5.2f}% ± {aggregate['train_200_em']['std']:<5.2f}% |")
    print(f"{'DEV Full-EM':<32} | {'21.67% ± 25.64%':<18} | {aggregate['dev_em']['mean']:>5.2f}% ± {aggregate['dev_em']['std']:<5.2f}% |")
    print(f"{'Test-Azar Full-EM':<32} | {'19.58% ± 22.84%':<18} | {aggregate['test_azar_em']['mean']:>5.2f}% ± {aggregate['test_azar_em']['std']:<5.2f}% |")
    print(f"{'Test-Comp Calibración ECE':<32} | {'No registrado':<18} | {aggregate['test_comp_ece']['mean']:>5.2f}% (S2: {aggregate['test_comp_stage2_ece']['mean']:>5.2f}%) |")
    print("=" * 85)
    print(f"VEREDICTO PUERTA DE SALIDA FASE 1: {verdict_str}")
    print(f"[*] Archivo consolidado guardado atómicamente con éxito en: {out_file}\n")


if __name__ == "__main__":
    main()
