"""
Suite de Paridad Numérica y Falsación Activa — Fase 2 (Hito 2.3):
1. Paridad numérica end-to-end:
   - Paridad estricta (<= 1e-4 en logits en las 4 etapas causales) frente a la referencia PyTorch add/sub.
   - 100% coincidencia textual en generate_causal_two_step frente a PyTorch FP32 y PyTorch add/sub
     sobre muestras reales de 'dev' y 'test-composicional'.
   - 0 pesos FP32 en RAM verificado en todas las proyecciones BitLinear.
2. Protocolo de falsación activa (3 variantes en tempfile aislado):
   - Falsación 1: Inversión de signos ternarios válidos (0 <-> 2, sin código 3) en payload de página 0.
   - Falsación 2: Alteración del factor de escala float32 scale_w (offset 12 de PagedHeader).
   - Falsación 3: Permutación de payloads entre páginas de idéntica geometría (pág 0 <-> pág 1).
   Verifica en las 3: max_logit_diff > 0.1 y degradación en la generación textual.
"""
import copy
import struct
import tempfile
import torch

from trace.models.causal import TRACECausalModel
from trace.layers.bitlinear import BitLinear
from trace.serialization.paged_bitcos import PagedBitCOSReader
from data import ProceduralCompositionalDataset


def count_fp32_weights_in_bitlinear(model: torch.nn.Module) -> int:
    total_fp32 = 0
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            if module.weight is not None and module.weight.dtype == torch.float32:
                total_fp32 += module.weight.numel()
            if hasattr(module, "cached_w_eff") and module.cached_w_eff is not None:
                if module.cached_w_eff.numel() > 0 and module.cached_w_eff.dtype == torch.float32:
                    total_fp32 += module.cached_w_eff.numel()
    return total_fp32


def test_numerical_parity_vs_pytorch_reference():
    torch.set_num_threads(2)
    ckpt_path = "checkpoints/trace_s456_fase1_bitnet_last.pt"
    bin_path = "checkpoints/trace_s456_fase1_paged_bitcos.bin"

    # 1. Cargar modelo de referencia PyTorch FP32
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_fp32 = ckpt["cfg"]
    cfg_fp32.device = "cpu"
    cfg_fp32.cpu_threads = 2

    model_ref_fp32 = TRACECausalModel(cfg_fp32)
    model_ref_fp32.load_state_dict(ckpt["model_state"], strict=True)
    model_ref_fp32.eval()

    # 2. Cargar modelo de referencia PyTorch add/sub
    cfg_addsub = copy.deepcopy(cfg_fp32)
    cfg_addsub.use_add_sub_inference = True

    model_ref_addsub = TRACECausalModel(cfg_addsub)
    model_ref_addsub.load_state_dict(ckpt["model_state"], strict=True)
    model_ref_addsub.eval()

    # 3. Cargar modelo Paged BitCOS v1 nativo (AVX2 + 0 pesos FP32 en RAM)
    cfg_paged = copy.deepcopy(cfg_fp32)
    cfg_paged.use_add_sub_inference = True

    model_paged = TRACECausalModel(cfg_paged)
    reader = PagedBitCOSReader(bin_path)
    reader.load_into_model(model_paged)
    reader.close()
    model_paged.eval()

    # Verificación explícita de 0 pesos FP32 en RAM para BitLinear
    fp32_count = count_fp32_weights_in_bitlinear(model_paged)
    assert fp32_count == 0, f"Error: detectados {fp32_count} pesos FP32 en BitLinear en model_paged."
    print(f"  [Auditoría RAM] Pesos FP32 en proyecciones BitLinear de model_paged: {fp32_count} (CERO estricto)")

    # 4. Evaluar paridad en muestras de 'dev' y 'test-composicional'
    for split_name in ("dev", "test-composicional"):
        dataset = ProceduralCompositionalDataset(cfg_fp32, split=split_name)
        num_eval = min(8, len(dataset.samples))
        stage_max_diffs_vs_addsub = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        stage_max_diffs_vs_fp32 = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        matched_text_count_fp32 = 0
        matched_text_count_addsub = 0

        for i in range(num_eval):
            prompt_b, _, _, _ = dataset.samples[i]

            # 4.1. Coincidencia en generate_causal_two_step
            with torch.no_grad():
                out_ref_fp32_str, _ = model_ref_fp32.generate_causal_two_step(prompt_b)
                out_ref_addsub_str, _ = model_ref_addsub.generate_causal_two_step(prompt_b)
                out_paged_str, _ = model_paged.generate_causal_two_step(prompt_b)

            if out_paged_str == out_ref_fp32_str:
                matched_text_count_fp32 += 1
            if out_paged_str == out_ref_addsub_str:
                matched_text_count_addsub += 1

            assert out_paged_str == out_ref_fp32_str, (
                f"Discrepancia textual en {split_name} muestra {i} frente a FP32:\n"
                f"Ref FP32: {out_ref_fp32_str}\n"
                f"Paged:    {out_paged_str}"
            )
            assert out_paged_str == out_ref_addsub_str, (
                f"Discrepancia textual en {split_name} muestra {i} frente a AddSub:\n"
                f"Ref AddSub: {out_ref_addsub_str}\n"
                f"Paged:      {out_paged_str}"
            )

            # 4.2. Medir max_logit_diff en las 4 etapas causales (stage_id in (0, 1, 2, 3))
            template_target = b"pw: ???\npms: ???\nwatt: ???\nms: ???\n"
            seq = [cfg_fp32.bos_token_id] + list(prompt_b) + list(template_target) + [cfg_fp32.eos_token_id]
            pad_len = max(0, cfg_fp32.max_seq_len - len(seq))
            full_seq = seq + [cfg_fp32.pad_token_id] * pad_len
            x = torch.tensor([full_seq], dtype=torch.long)
            t = torch.ones(1)

            for s_id in (0, 1, 2, 3):
                with torch.no_grad():
                    logits_fp32 = model_ref_fp32.forward_backbone(x, t, stage_id=s_id)
                    logits_addsub = model_ref_addsub.forward_backbone(x, t, stage_id=s_id)
                    logits_paged = model_paged.forward_backbone(x, t, stage_id=s_id)

                diff_addsub = float((logits_addsub - logits_paged).abs().max().item())
                diff_fp32 = float((logits_fp32 - logits_paged).abs().max().item())

                if diff_addsub > stage_max_diffs_vs_addsub[s_id]:
                    stage_max_diffs_vs_addsub[s_id] = diff_addsub
                if diff_fp32 > stage_max_diffs_vs_fp32[s_id]:
                    stage_max_diffs_vs_fp32[s_id] = diff_fp32

        print(f"  [{split_name.upper()}] Exactitud textual vs PyTorch FP32:   {matched_text_count_fp32}/{num_eval} (100.0%)")
        print(f"  [{split_name.upper()}] Exactitud textual vs PyTorch AddSub: {matched_text_count_addsub}/{num_eval} (100.0%)")
        print(f"  [{split_name.upper()}] Discrepancia máxima de logits vs PyTorch AddSub (kernel AVX2 bit a bit):")
        for s_id in (0, 1, 2, 3):
            print(f"    - Etapa {s_id}: max_logit_diff = {stage_max_diffs_vs_addsub[s_id]:.6e} (vs FP32: {stage_max_diffs_vs_fp32[s_id]:.4f})")
            assert stage_max_diffs_vs_addsub[s_id] <= 1e-4, (
                f"Violación de paridad en {split_name} Etapa {s_id}: {stage_max_diffs_vs_addsub[s_id]} > 1e-4"
            )


def test_falsification_battery():
    """Protocolo de Falsación Activa en 3 variantes (aislado en tempfile)."""
    torch.set_num_threads(2)
    bin_path = "checkpoints/trace_s456_fase1_paged_bitcos.bin"
    ckpt_path = "checkpoints/trace_s456_fase1_bitnet_last.pt"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    cfg.device = "cpu"
    cfg.cpu_threads = 2
    cfg_paged = copy.deepcopy(cfg)
    cfg_paged.use_add_sub_inference = True

    # Modelo de referencia no alterado
    model_sano = TRACECausalModel(cfg_paged)
    reader_sano = PagedBitCOSReader(bin_path)
    reader_sano.load_into_model(model_sano)
    reader_sano.close()
    model_sano.eval()

    with open(bin_path, "rb") as f:
        canonical_bytes = bytearray(f.read())

    reader = PagedBitCOSReader(bin_path)
    page_sz = reader.page_size
    pgs_off = reader.pages_offset
    page0_off = pgs_off
    page1_off = pgs_off + page_sz
    reader.close()

    dataset = ProceduralCompositionalDataset(cfg, split="dev")
    prompt_b, _, _, _ = dataset.samples[0]
    template_target = b"pw: ???\npms: ???\nwatt: ???\nms: ???\n"
    seq = [cfg.bos_token_id] + list(prompt_b) + list(template_target) + [cfg.eos_token_id]
    pad_len = max(0, cfg.max_seq_len - len(seq))
    full_seq = seq + [cfg.pad_token_id] * pad_len
    x = torch.tensor([full_seq], dtype=torch.long)
    t = torch.ones(1)

    with torch.no_grad():
        out_sano_str, _ = model_sano.generate_causal_two_step(prompt_b)
        logits_sano = model_sano.forward_backbone(x, t, stage_id=0)

    print(f"  [Línea Base Sana] Generación: {out_sano_str.strip()}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # -----------------------------------------------------------------
        # Falsación 1: Inversión de signos válidos (0 <-> 2) en payload pág 0
        # -----------------------------------------------------------------
        f1_bytes = bytearray(canonical_bytes)
        hdr_sz = 28  # struct <IIHHfII
        payload_start = page0_off + hdr_sz
        payload_end = page0_off + page_sz

        # Invertir todos los bytes de payload de página 0
        for idx in range(payload_start, payload_end):
            b = f1_bytes[idx]
            c0, c1, c2, c3 = (b & 3), ((b >> 2) & 3), ((b >> 4) & 3), ((b >> 6) & 3)
            # Inversión matemática sin generar código 3: 0 -> 2, 2 -> 0, 1 -> 1
            inv_c0 = (2 - c0) if c0 <= 2 else c0
            inv_c1 = (2 - c1) if c1 <= 2 else c1
            inv_c2 = (2 - c2) if c2 <= 2 else c2
            inv_c3 = (2 - c3) if c3 <= 2 else c3
            f1_bytes[idx] = inv_c0 | (inv_c1 << 2) | (inv_c2 << 4) | (inv_c3 << 6)

        f1_path = f"{tmpdir}/falsified_1_sign_flip.bin"
        with open(f1_path, "wb") as f:
            f.write(f1_bytes)

        m1 = TRACECausalModel(cfg_paged)
        r1 = PagedBitCOSReader(f1_path)
        r1.load_into_model(m1)
        r1.close()
        m1.eval()

        with torch.no_grad():
            logits_f1 = m1.forward_backbone(x, t, stage_id=0)
            out_f1_str, _ = m1.generate_causal_two_step(prompt_b)

        diff1 = float((logits_sano - logits_f1).abs().max().item())
        print(f"  [Falsación 1 - Sign Flip (0 <-> 2)] max_logit_diff = {diff1:.4f}")
        print(f"    Texto alterado: {out_f1_str.strip()}")
        assert diff1 > 0.1, f"Falsación 1 fallida: diff={diff1} <= 0.1 (sin efecto en logits)"
        assert out_sano_str != out_f1_str, "Falsación 1 fallida: texto generado idéntico tras alterar pesos."

        # -----------------------------------------------------------------
        # Falsación 2: Alteración de scale_w (float32 en offset 12)
        # -----------------------------------------------------------------
        f2_bytes = bytearray(canonical_bytes)
        scale_offset = page0_off + 12
        orig_scale = struct.unpack_from("<f", f2_bytes, scale_offset)[0]
        # Multiplicar scale_w por 10.0
        struct.pack_into("<f", f2_bytes, scale_offset, orig_scale * 10.0)

        f2_path = f"{tmpdir}/falsified_2_scale_w.bin"
        with open(f2_path, "wb") as f:
            f.write(f2_bytes)

        m2 = TRACECausalModel(cfg_paged)
        r2 = PagedBitCOSReader(f2_path)
        r2.load_into_model(m2)
        r2.close()
        m2.eval()

        with torch.no_grad():
            logits_f2 = m2.forward_backbone(x, t, stage_id=0)
            out_f2_str, _ = m2.generate_causal_two_step(prompt_b)

        diff2 = float((logits_sano - logits_f2).abs().max().item())
        print(f"  [Falsación 2 - Scale Alteration] max_logit_diff = {diff2:.4f} (escala orig={orig_scale:.4e})")
        print(f"    Texto alterado: {out_f2_str.strip()}")
        assert diff2 > 0.1, f"Falsación 2 fallida: diff={diff2} <= 0.1 (sin efecto en logits)"
        assert out_sano_str != out_f2_str, "Falsación 2 fallida: texto generado idéntico tras alterar escala."

        # -----------------------------------------------------------------
        # Falsación 3: Permutación de payloads entre dos páginas de igual geometría
        # -----------------------------------------------------------------
        f3_bytes = bytearray(canonical_bytes)
        # Intercambiar los payloads completos entre página 0 y página 1
        p0_payload = bytes(f3_bytes[page0_off + hdr_sz : page0_off + page_sz])
        p1_payload = bytes(f3_bytes[page1_off + hdr_sz : page1_off + page_sz])
        f3_bytes[page0_off + hdr_sz : page0_off + page_sz] = p1_payload
        f3_bytes[page1_off + hdr_sz : page1_off + page_sz] = p0_payload

        f3_path = f"{tmpdir}/falsified_3_page_permute.bin"
        with open(f3_path, "wb") as f:
            f.write(f3_bytes)

        m3 = TRACECausalModel(cfg_paged)
        r3 = PagedBitCOSReader(f3_path)
        r3.load_into_model(m3)
        r3.close()
        m3.eval()

        with torch.no_grad():
            logits_f3 = m3.forward_backbone(x, t, stage_id=0)
            out_f3_str, _ = m3.generate_causal_two_step(prompt_b)

        diff3 = float((logits_sano - logits_f3).abs().max().item())
        print(f"  [Falsación 3 - Page Permutation] max_logit_diff = {diff3:.4f}")
        print(f"    Texto alterado: {out_f3_str.strip()}")
        assert diff3 > 0.1, f"Falsación 3 fallida: diff={diff3} <= 0.1 (sin efecto en logits)"
        assert out_sano_str != out_f3_str, "Falsación 3 fallida: texto generado idéntico tras permutar páginas."


if __name__ == "__main__":
    print("=" * 80)
    print("Batería de Paridad Numérica y Falsación Activa — Basic-Lite TRACE")
    print("=" * 80)
    print("\n1. PARIDAD NUMÉRICA EN 4 ETAPAS (DEV Y TEST-COMPOSICIONAL):")
    test_numerical_parity_vs_pytorch_reference()
    print("✓ Paridad numérica estricta <= 1e-4 y coincidencia textual 100% superada.")

    print("\n2. BATERÍA DE FALSACIÓN ACTIVA (3 PRUEBAS IRREFUTABLES):")
    test_falsification_battery()
    print("✓ Las 3 falsaciones demostraron colapso determinista (diff > 0.1 y texto degradado).")
    print("✓ Hito 2.3 completado exitosamente.")
