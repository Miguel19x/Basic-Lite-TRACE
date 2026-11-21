#!/usr/bin/env python3
# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Script de consolidación, evaluación rigurosa y reporte comparativo de 5 semillas.
Carga los checkpoints generados (trace_s42, trace_s123, trace_s456, trace_s789, trace_s999),
evalúa exhaustivamente en DEV (108) y TEST-COMP (480), calcula estadísticas agregadas
con t-Student (CI95) y contrasta contra la línea base histórica.
"""

import os
import json
import time
import numpy as np
import torch

from config import HybridConfig
from data import ProceduralCompositionalDataset
from trace.models.causal import TRACECausalModel
from eval import evaluate_causal


def compute_ci95(vals):
    n = len(vals)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(vals))
    if n == 1:
        return mean, 0.0, 0.0
    std = float(np.std(vals, ddof=1))
    # t crítico para 95% bilateral: df=4 -> 2.776, df=3 -> 3.182, etc.
    t_crit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 1.96)
    ci95 = float(t_crit * (std / np.sqrt(n)))
    return mean, std, ci95


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [42, 123, 456, 789, 999]

    print("=" * 85)
    print("Basic-Lite TRACE — Consolidación y Auditoría Confirmatoria de 5 Semillas")
    print(f"Dispositivo: {device} | Semillas objetivo: {seeds}")
    print("=" * 85)

    cfg = HybridConfig(device=str(device))
    dev_dataset = ProceduralCompositionalDataset(cfg, split="dev")
    test_comp_dataset = ProceduralCompositionalDataset(cfg, split="test-composicional")

    per_seed_results = {}

    for s in seeds:
        ckpt_path = f"trace_s{s}_checkpoint.pt"
        if not os.path.exists(ckpt_path):
            # Alternativa por si se guardó con nombre por defecto
            if s == 42 and os.path.exists("trace_checkpoint.pt"):
                ckpt_path = "trace_checkpoint.pt"
            else:
                print(f"[!] AVISO: No se encontró el checkpoint para semilla {s}: {ckpt_path}")
                continue

        print(f"\n---> Cargando y evaluando Semilla {s} desde: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = TRACECausalModel(cfg).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        t0 = time.time()
        dev_res = evaluate_causal(model, dev_dataset, device)
        test_res = evaluate_causal(model, test_comp_dataset, device)
        dt = time.time() - t0

        per_seed_results[str(s)] = {
            "dev": dev_res,
            "test_comp": test_res,
            "eval_time_sec": dt,
        }

        print(
            f"     DEV: ExactMatch={dev_res['exact_match_pct']:.2f}% ({dev_res['em_count']}/{dev_res['total_cases']}) | "
            f"ScratchpadEM={dev_res['scratchpad_exact_pct']:.2f}% | DigAcc={dev_res['digit_acc_pct']:.2f}%"
        )
        print(
            f"     TEST-COMP: ExactMatch={test_res['exact_match_pct']:.2f}% ({test_res['em_count']}/{test_res['total_cases']}) | "
            f"ScratchpadEM={test_res['scratchpad_exact_pct']:.2f}% | DigAcc={test_res['digit_acc_pct']:.2f}%"
        )

    if not per_seed_results:
        print("\n[ERROR] No se pudo evaluar ningún checkpoint. Comprueba las rutas en el entorno.")
        return

    # Extraer arrays para estadísticas
    test_em_vals = [r["test_comp"]["exact_match_pct"] for r in per_seed_results.values()]
    test_sp_vals = [r["test_comp"]["scratchpad_exact_pct"] for r in per_seed_results.values()]
    test_dig_vals = [r["test_comp"]["digit_acc_pct"] for r in per_seed_results.values()]
    dev_em_vals = [r["dev"]["exact_match_pct"] for r in per_seed_results.values()]
    dev_sp_vals = [r["dev"]["scratchpad_exact_pct"] for r in per_seed_results.values()]

    m_em, s_em, ci_em = compute_ci95(test_em_vals)
    m_sp, s_sp, ci_sp = compute_ci95(test_sp_vals)
    m_dig, s_dig, ci_dig = compute_ci95(test_dig_vals)
    m_dev_em, s_dev_em, ci_dev_em = compute_ci95(dev_em_vals)
    m_dev_sp, s_dev_sp, ci_dev_sp = compute_ci95(dev_sp_vals)

    # Cargar línea base histórica si existe
    baseline_path = "results/eval_confirmatorio_5semillas_results.json"
    has_baseline = os.path.exists(baseline_path)
    b_em, b_ci_em, b_sp, b_ci_sp = 5.122, 6.059, 81.876, 12.585
    if has_baseline:
        try:
            with open(baseline_path, "r", encoding="utf-8") as f:
                b_data = json.load(f)
                b_em = b_data["aggregate_summary"]["test_comp_exact_match_pct"]["mean"]
                b_ci_em = b_data["aggregate_summary"]["test_comp_exact_match_pct"]["ci95"]
                b_sp = b_data["aggregate_summary"]["test_comp_scratchpad_exact_pct"]["mean"]
                b_ci_sp = b_data["aggregate_summary"]["test_comp_scratchpad_exact_pct"]["ci95"]
        except Exception:
            pass

    print("\n" + "=" * 85)
    print("TABLA COMPARATIVA DETALLADA POR SEMILLA")
    print("=" * 85)
    print(f"{'Semilla':<10} | {'DEV EM':<10} | {'DEV SP-EM':<12} | {'TEST EM':<10} | {'TEST SP-EM':<12} | {'TEST DigAcc':<12}")
    print("-" * 85)
    for s_str, r in per_seed_results.items():
        print(
            f"{s_str:<10} | "
            f"{r['dev']['exact_match_pct']:>6.2f}%    | "
            f"{r['dev']['scratchpad_exact_pct']:>8.2f}%    | "
            f"{r['test_comp']['exact_match_pct']:>6.2f}%    | "
            f"{r['test_comp']['scratchpad_exact_pct']:>8.2f}%    | "
            f"{r['test_comp']['digit_acc_pct']:>8.2f}%"
        )
    print("=" * 85)

    print("\n" + "=" * 85)
    print("RESUMEN ESTADÍSTICO AGREGADO (5 SEMILLAS — CONFIRMATORIO)")
    print("=" * 85)
    print(f"{'Métrica':<35} | {'Histórico (c874140)':<22} | {'Nuevo (exp/unblock)':<22}")
    print("-" * 85)
    print(f"{'TEST-COMP Exact Match (Full-EM)':<35} | {b_em:>5.2f}% ± {b_ci_em:<6.2f}%       | {m_em:>5.2f}% ± {ci_em:<6.2f}%")
    print(f"{'TEST-COMP Scratchpad-EM':<35} | {b_sp:>5.2f}% ± {b_ci_sp:<6.2f}%       | {m_sp:>5.2f}% ± {ci_sp:<6.2f}%")
    print(f"{'TEST-COMP Exactitud Dígitos':<35} | {'81.97% ± 3.90%':<22} | {m_dig:>5.2f}% ± {ci_dig:<6.2f}%")
    print(f"{'DEV Exact Match':<35} | {'7.78% ± 8.60%':<22} | {m_dev_em:>5.2f}% ± {ci_dev_em:<6.2f}%")
    print(f"{'DEV Scratchpad-EM':<35} | {'83.52% ± 12.76%':<22} | {m_dev_sp:>5.2f}% ± {ci_dev_sp:<6.2f}%")
    print("=" * 85)

    # Exportar JSON oficial
    out_payload = {
        "experiment": "TRACE_Causal_2Stage_Confirmatorio_5Semillas_Unblocked",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "seeds_evaluated": [int(s) for s in per_seed_results.keys()],
        "per_seed_results": per_seed_results,
        "aggregate_summary": {
            "test_comp_exact_match_pct": {
                "mean": m_em, "std": s_em, "ci95": ci_em, "values": test_em_vals
            },
            "test_comp_scratchpad_exact_pct": {
                "mean": m_sp, "std": s_sp, "ci95": ci_sp, "values": test_sp_vals
            },
            "dev_exact_match_pct": {
                "mean": m_dev_em, "std": s_dev_em, "ci95": ci_dev_em, "values": dev_em_vals
            },
            "dev_scratchpad_exact_pct": {
                "mean": m_dev_sp, "std": s_dev_sp, "ci95": ci_dev_sp, "values": dev_sp_vals
            },
        },
    }

    out_file = "results/eval_confirmatorio_unblock_5semillas_results.json"
    os.makedirs("results", exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)
    print(f"\n[*] Informe consolidado exportado a: {out_file}")


if __name__ == "__main__":
    main()
