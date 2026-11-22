#!/usr/bin/env python3
# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Celda de Diagnóstico y Auditoría Causal de 5 Semillas para Google Colab (Fase 2b).
Implementa:
- Muestreo Train-200 sin reemplazo (§1.2.4).
- Auditoría de las 5 semillas (Train-200, DEV, Test-Azar, Test-Comp).
- Desglose por acarreo/préstamo real (delta = aw + ow) y por par reservado.
- Exportación formal a eval_fase2b_unblock_5semillas_results.json con bootstrap percentil.
"""

import os, sys, json, random
import numpy as np
from collections import Counter, defaultdict

if "/content/Basic-Lite-TRACE" not in sys.path:
    sys.path.insert(0, "/content/Basic-Lite-TRACE")
current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

if "trace" in sys.modules and not hasattr(sys.modules["trace"], "__path__"):
    del sys.modules["trace"]

import torch
from config import HybridConfig
from data import ProceduralCompositionalDataset, parse_fixed_scratchpad
from trace.models.causal import TRACECausalModel
from eval import evaluate_causal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = HybridConfig(device=str(device))

# Lookup exacto de operandos desde data.py
NODO_MAP = {n: (bw, bms) for n, bw, bms in ProceduralCompositionalDataset.NODOS}
CARGA_MAP = {c: (mw, mms) for c, mw, mms in ProceduralCompositionalDataset.CARGAS}
RED_MAP = {r: (aw, ams) for r, aw, ams in ProceduralCompositionalDataset.REDUNDANCIAS}
PERF_MAP = {p: (ow, oms) for p, ow, oms in ProceduralCompositionalDataset.PERFILES}

def parse_prompt_fields(prompt_txt: str):
    kv = {}
    for line in prompt_txt.strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            kv[k.strip()] = v.strip()
    return kv

def has_carry_or_borrow(sub: int, delta: int):
    if delta < 0:
        return "borrow" if (sub % 10) < (abs(delta) % 10) or (sub % 100) < (abs(delta) % 100) else "sub_no_borrow"
    u_carry = ((sub % 10) + (delta % 10)) >= 10
    d_carry = ((sub % 100) + (delta % 100)) >= 100
    if u_carry and d_carry:
        return "double_carry"
    if u_carry or d_carry:
        return "single_carry"
    return "no_carry"

# 1. Particiones exactas (muestra Train-200 SIN reemplazo, §1.2.4)
train_full_ds = ProceduralCompositionalDataset(cfg, split="train")
train_200_ds = ProceduralCompositionalDataset(cfg, split="train")
train_200_ds.samples = random.Random(42).sample(train_full_ds.unique_records, 200)

dev_ds = ProceduralCompositionalDataset(cfg, split="dev")
test_azar_ds = ProceduralCompositionalDataset(cfg, split="test-azar")
test_comp_ds = ProceduralCompositionalDataset(cfg, split="test-composicional")

# 2. Auditoría de las 5 semillas (Train-200, DEV, Test-Azar, Test-Comp)
seeds = [42, 123, 456, 789, 999]
per_seed = {}

print("=" * 92)
print(f"{'Semilla':<8} | {'Train-200 EM':>12} | {'DEV EM':>8} | {'Test-Azar EM':>12} | {'Test-Comp EM':>12} | {'Comp SP-EM':>10}")
print("-" * 92)

for s in seeds:
    ckpt_path = f"trace_s{s}_checkpoint.pt"
    if not os.path.exists(ckpt_path):
        print(f"Semilla {s:<3}: checkpoint no encontrado ({ckpt_path})")
        continue
    m = TRACECausalModel(cfg).to(device)
    m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model_state"])
    m.eval()

    r_tr = evaluate_causal(m, train_200_ds, device)
    r_dv = evaluate_causal(m, dev_ds, device)
    r_az = evaluate_causal(m, test_azar_ds, device)
    r_cp = evaluate_causal(m, test_comp_ds, device)

    per_seed[str(s)] = {
        "train_200_unique": r_tr,
        "dev": r_dv,
        "test_azar": r_az,
        "test_comp": r_cp,
    }
    print(
        f"{s:<8} | {r_tr['exact_match_pct']:>11.2f}% | {r_dv['exact_match_pct']:>7.2f}% | "
        f"{r_az['exact_match_pct']:>11.2f}% | {r_cp['exact_match_pct']:>11.2f}% | {r_cp['scratchpad_exact_pct']:>9.2f}%"
    )

# Bootstrap percentil (10,000 repeticiones sobre semillas disponibles)
def summarize_metric(vals):
    arr = np.array(vals, dtype=float)
    rng = np.random.default_rng(42)
    boots = rng.choice(arr, size=(10000, len(arr)), replace=True).mean(axis=1)
    return {
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 4),
        "median": round(float(np.median(arr)), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "ci95_bootstrap": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
        "values": [round(float(v), 4) for v in arr],
    }

# 3. Radiografía detallada de los fallos de la Semilla 999 en Test-Composicional
ckpt_999_path = "trace_s999_checkpoint.pt"
if os.path.exists(ckpt_999_path):
    m999 = TRACECausalModel(cfg).to(device)
    m999.load_state_dict(torch.load(ckpt_999_path, map_location=device, weights_only=False)["model_state"])
    m999.eval()

    pair_stats = defaultdict(lambda: [0, 0])   # (perfil, carga) -> [ok, total]
    carry_w_stats = defaultdict(lambda: [0, 0])
    carry_ms_stats = defaultdict(lambda: [0, 0])
    col_err = Counter()
    diff_bucket = Counter()
    field_fail = Counter()
    sp_fail = 0
    total_err = 0

    for prompt_b, target_b, exp_w, exp_ms in test_comp_ds.samples:
        sub_w, sub_ms, _, _ = test_comp_ds.record_meta[(prompt_b, target_b)]
        kv = parse_prompt_fields(prompt_b.decode("utf-8"))
        aw, ams = RED_MAP[kv["red"]]
        ow, oms = PERF_MAP[kv["perfil"]]
        dw, dms = aw + ow, ams + oms

        cat_w = has_carry_or_borrow(sub_w, dw)
        cat_ms = has_carry_or_borrow(sub_ms, dms)
        pair_key = f"{kv['perfil']} x {kv['carga']}"

        gen_str, _ = m999.generate_causal_two_step(prompt_b)
        valid, pw, pms, w, ms = parse_fixed_scratchpad(gen_str)

        w_ok = valid and (w == exp_w)
        ms_ok = valid and (ms == exp_ms)
        sp_ok = valid and (pw == sub_w) and (pms == sub_ms)
        full_ok = sp_ok and w_ok and ms_ok

        pair_stats[pair_key][1] += 1
        pair_stats[pair_key][0] += int(full_ok)
        carry_w_stats[cat_w][1] += 1
        carry_w_stats[cat_w][0] += int(w_ok)
        carry_ms_stats[cat_ms][1] += 1
        carry_ms_stats[cat_ms][0] += int(ms_ok)

        if not full_ok:
            total_err += 1
            if not sp_ok:
                sp_fail += 1
            if not w_ok and not ms_ok:
                field_fail["ambos (watt y ms)"] += 1
            elif not w_ok:
                field_fail["solo watt"] += 1
            elif not ms_ok:
                field_fail["solo ms"] += 1
            else:
                field_fail["solo scratchpad"] += 1

            if valid:
                for name, pred_v, exp_v in [("watt", w, exp_w), ("ms", ms, exp_ms)]:
                    if pred_v != exp_v:
                        ps, es = f"{pred_v:03d}", f"{exp_v:03d}"
                        if ps[0] != es[0]: col_err[f"{name}_centenas"] += 1
                        if ps[1] != es[1]: col_err[f"{name}_decenas"] += 1
                        if ps[2] != es[2]: col_err[f"{name}_unidades"] += 1
                        d = abs(pred_v - exp_v)
                        if d == 1: diff_bucket[f"{name}_±1"] += 1
                        elif d == 10: diff_bucket[f"{name}_±10 (acarreo D)"] += 1
                        elif d == 100: diff_bucket[f"{name}_±100 (acarreo C)"] += 1
                        elif d % 10 == 0: diff_bucket[f"{name}_multiplo_10"] += 1
                        else: diff_bucket[f"{name}_otro"] += 1

    print("\n" + "=" * 92)
    print(f"RADIOGRAFÍA DE LOS {total_err} FALLOS — SEMILLA 999 (TEST-COMPOSICIONAL, N=480)")
    print("=" * 92)
    print(f"Errores originados en Etapa 1 (SP-EM fallido): {sp_fail} / {total_err}")
    print("Desglose por campo fallido:", dict(field_fail))
    print("Errores por columna decimal:", dict(col_err))
    print("Tipo de error numérico:", dict(diff_bucket))

    print("\nExact Match por par reservado (perfil x carga):")
    for k, (ok, tot) in sorted(pair_stats.items(), key=lambda x: x[1][0]/x[1][1]):
        print(f"  {k:<22}: {ok:>2}/{tot} ({ok/tot*100:5.1f}%)")

    print("\nExactitud de WATT por tipo de acarreo/préstamo (delta = a_w + o_w):")
    for k, (ok, tot) in sorted(carry_w_stats.items()):
        print(f"  {k:<16}: {ok:>3}/{tot} ({ok/tot*100:5.1f}%)")

    print("\nExactitud de MS por tipo de acarreo/préstamo (delta = a_ms + o_ms):")
    for k, (ok, tot) in sorted(carry_ms_stats.items()):
        print(f"  {k:<16}: {ok:>3}/{tot} ({ok/tot*100:5.1f}%)")

# 4. Guardar JSON exploratorio completo (Fase 2b)
if per_seed:
    os.makedirs("results", exist_ok=True)
    out_json = {
        "experiment": "TRACE_Fase2b_Exploratorio_Unblock_5Semillas",
        "classification": "EXPLORATORIO (post-prereg-v1, §9.4)",
        "branch": "exp/arithmetic-unblock",
        "seeds": [int(k) for k in per_seed.keys()],
        "per_seed_results": per_seed,
        "aggregate_summary": {
            "train_200_em": summarize_metric([v["train_200_unique"]["exact_match_pct"] for v in per_seed.values()]),
            "dev_em": summarize_metric([v["dev"]["exact_match_pct"] for v in per_seed.values()]),
            "test_azar_em": summarize_metric([v["test_azar"]["exact_match_pct"] for v in per_seed.values()]),
            "test_comp_em": summarize_metric([v["test_comp"]["exact_match_pct"] for v in per_seed.values()]),
            "test_comp_sp_em": summarize_metric([v["test_comp"]["scratchpad_exact_pct"] for v in per_seed.values()]),
        },
    }
    with open("results/eval_fase2b_unblock_5semillas_results.json", "w") as f:
        json.dump(out_json, f, indent=2)
    print("\n✓ Exportado: results/eval_fase2b_unblock_5semillas_results.json")
