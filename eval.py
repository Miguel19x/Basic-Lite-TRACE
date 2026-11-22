# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

import time
from trace.models.causal import TRACECausalModel
from typing import Any

import torch

from data import ProceduralCompositionalDataset, parse_fixed_scratchpad


def compute_ece(confidences: list[float], correctness: list[float], num_bins: int = 10) -> float:
    """Calcula Expected Calibration Error (ECE) ponderado en porcentaje."""
    if not confidences or len(confidences) != len(correctness):
        return 0.0
    bin_limits = [i / float(num_bins) for i in range(num_bins + 1)]
    N = float(len(confidences))
    ece = 0.0
    for b in range(num_bins):
        low, high = bin_limits[b], bin_limits[b + 1]
        bin_mask = [(low <= c < high) if b < num_bins - 1 else (low <= c <= high) for c in confidences]
        bin_count = sum(bin_mask)
        if bin_count > 0:
            bin_acc = sum(corr for corr, m in zip(correctness, bin_mask) if m) / float(bin_count)
            bin_conf = sum(conf for conf, m in zip(confidences, bin_mask) if m) / float(bin_count)
            ece += (bin_count / N) * abs(bin_acc - bin_conf)
    return ece * 100.0


def evaluate_causal(
    model: TRACECausalModel,
    dataset: ProceduralCompositionalDataset,
    device: torch.device,
) -> dict[str, Any]:
    """
    Evaluación rigurosa del modelo TRACE Causal (Basic-Lite TRACE) bajo inferencia de 2 pasos.
    Calcula Exact Match Total, Scratchpad EM, Final EM, Exactitud de Dígitos,
    Validez de Formato, Latencia y Calibración ECE (Global y Etapa 2).
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

    samples = dataset.samples

    for idx in range(total_cases):
        prompt_b, target_b, expected_watt, expected_ms = samples[idx]
        sub_w, sub_ms, _, _ = dataset.record_meta.get(
            (prompt_b, target_b), (None, None, expected_watt, expected_ms)
        )

        t0 = time.perf_counter()
        gen_str, meta = model.generate_causal_two_step(prompt_b)
        t1 = time.perf_counter()
        total_time_ms += (t1 - t0) * 1000.0

        valid, pred_pw, pred_pms, pred_watt, pred_ms = parse_fixed_scratchpad(gen_str)

        if valid:
            fmt_valid_count += 1

            # Comparación de Subtotales (Etapa 1)
            sp_match = (pred_pw == sub_w) and (pred_pms == sub_ms)
            if sp_match:
                sp_em_count += 1

            # Comparación de Totales Finales (Etapa 2)
            fn_match = (pred_watt == expected_watt) and (pred_ms == expected_ms)
            if fn_match:
                final_em_count += 1

            # Exact Match Total
            if sp_match and fn_match:
                em_total_count += 1

        # Comparación granular de dígitos
        expected_target_str = target_b.decode("utf-8")
        # Ambos deben tener los 12 dígitos en sus posiciones
        # Target: pw: XXX\npms: YYY\nwatt: ZZZ\nms: WWW\n
        digit_indices = [
            (4, 7),   # pw (dígitos 0, 1, 2)
            (13, 16), # pms (dígitos 3, 4, 5)
            (23, 26), # watt (dígitos 6, 7, 8 - Etapa 2)
            (31, 34), # ms (dígitos 9, 10, 11 - Etapa 2)
        ]

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

    # Calibración desacoplada
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

    return {
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
