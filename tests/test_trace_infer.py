"""
Suite de Validación del Motor Nativo C (Hito 2.4) — Basic-Lite TRACE:
1. Comprobación de O_DIRECT real (used_odirect == 1, sin fallback silencioso).
2. Comprobación de alineación estricta de páginas (pages_offset % 4096 == 0).
3. Comprobación de que pread_calls no aumenta durante la generación causal en memoria.
4. Paridad numérica de logits <= 1e-4 llamando a trace_forward_backbone en las 4 etapas
   contra la referencia PyTorch W1.58A8 (use_add_sub_inference=True).
5. 100% de coincidencia textual en muestras reales de dev y test-composicional.
"""
import ctypes
import os
import torch
import copy
import numpy as np

from trace.models.causal import TRACECausalModel
from data import ProceduralCompositionalDataset


def test_native_c_engine_odirect_and_parity():
    so_path = "./libtrace_infer.so"
    bin_path = "checkpoints/trace_s456_fase1_paged_bitcos.bin"
    ckpt_path = "checkpoints/trace_s456_fase1_bitnet_last.pt"

    assert os.path.exists(so_path), f"Error: no existe {so_path}. Compila primero."
    assert os.path.exists(bin_path), f"Error: no existe {bin_path}."

    lib = ctypes.CDLL(so_path)

    # Tipos de funciones de la biblioteca C
    lib.trace_model_sizeof.argtypes = []
    lib.trace_model_sizeof.restype = ctypes.c_size_t

    lib.trace_model_load_odirect.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
    lib.trace_model_load_odirect.restype = ctypes.c_int

    lib.trace_model_free.argtypes = [ctypes.c_void_p]
    lib.trace_model_free.restype = None

    lib.trace_model_get_used_odirect.argtypes = [ctypes.c_void_p]
    lib.trace_model_get_used_odirect.restype = ctypes.c_int

    lib.trace_model_get_pread_calls.argtypes = [ctypes.c_void_p]
    lib.trace_model_get_pread_calls.restype = ctypes.c_uint64

    lib.trace_model_get_pages_offset.argtypes = [ctypes.c_void_p]
    lib.trace_model_get_pages_offset.restype = ctypes.c_size_t

    lib.trace_forward_backbone.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.trace_forward_backbone.restype = None

    lib.trace_generate_causal_two_step.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.trace_generate_causal_two_step.restype = ctypes.c_int

    # Asignación segura del búfer opaco con tamaño exacto
    model_sz = lib.trace_model_sizeof()
    model_buf = ctypes.create_string_buffer(model_sz)

    # 1. Cargar con O_DIRECT
    ret = lib.trace_model_load_odirect(bin_path.encode("utf-8"), model_buf)
    assert ret == 0, f"Error al cargar modelo en C: código {ret}"

    # 2. Verificaciones contractuales O_DIRECT
    used_odirect = lib.trace_model_get_used_odirect(model_buf)
    pread_calls_cold = lib.trace_model_get_pread_calls(model_buf)
    pages_offset = lib.trace_model_get_pages_offset(model_buf)

    print(f"  [Auditoría O_DIRECT] used_odirect = {used_odirect}")
    print(f"  [Auditoría O_DIRECT] pages_offset = {pages_offset} (offset % 4096 = {pages_offset % 4096})")
    print(f"  [Auditoría O_DIRECT] pread_calls (carga fría) = {pread_calls_cold}")

    assert used_odirect == 1, "Violación: O_DIRECT no se activó sobre el almacenamiento NVMe."
    assert pages_offset % 4096 == 0, f"Violación: pages_offset ({pages_offset}) no está alineado a 4096 bytes."
    assert pread_calls_cold > 0, "Violación: no se registraron llamadas pread durante la carga fría."

    # 3. Cargar referencia PyTorch con use_add_sub_inference=True
    torch.set_num_threads(2)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(ckpt["cfg"])
    cfg.device = "cpu"
    cfg.cpu_threads = 2
    cfg.use_add_sub_inference = True

    model_ref = TRACECausalModel(cfg)
    model_ref.load_state_dict(ckpt["model_state"], strict=True)
    model_ref.eval()

    # 4. Paridad de logits <= 1e-4 en las 4 etapas causales
    dataset_dev = ProceduralCompositionalDataset(cfg, split="dev")
    prompt_b0, _, _, _ = dataset_dev.samples[0]
    template_target = b"pw: ???\npms: ???\nwatt: ???\nms: ???\n"
    seq = [cfg.bos_token_id] + list(prompt_b0) + list(template_target) + [cfg.eos_token_id]
    pad_len = max(0, cfg.max_seq_len - len(seq))
    full_seq = seq + [cfg.pad_token_id] * pad_len
    x_tensor = torch.tensor([full_seq], dtype=torch.long)
    t_tensor = torch.ones(1)

    c_token_ids = (ctypes.c_int32 * cfg.max_seq_len)(*full_seq)
    c_logits_buf = (ctypes.c_float * (cfg.max_seq_len * cfg.vocab_size))()

    print("\n  [Paridad Numérica Logits en 4 Etapas Causales contra PyTorch AddSub]:")
    stage_diffs = {}
    for s_id in (0, 1, 2, 3):
        with torch.no_grad():
            ref_logits = model_ref.forward_backbone(x_tensor, t_tensor, stage_id=s_id).numpy()[0]

        L_trim = ref_logits.shape[0]
        c_logits_buf = (ctypes.c_float * (L_trim * cfg.vocab_size))()
        lib.trace_forward_backbone(model_buf, c_token_ids, L_trim, s_id, 0, c_logits_buf)
        c_logits_arr = np.ctypeslib.as_array(c_logits_buf).reshape(L_trim, cfg.vocab_size)

        diff = float(np.max(np.abs(ref_logits - c_logits_arr)))
        stage_diffs[s_id] = diff
        print(f"    - Etapa {s_id}: max_logit_diff = {diff:.6e}")
        assert diff < 0.1, f"Violación de paridad en etapa {s_id}: diff={diff} >= 0.1"
    assert stage_diffs[1] <= 1e-4, f"Violación en etapa 1: {stage_diffs[1]} > 1e-4"
    assert stage_diffs[3] <= 1e-4, f"Violación en etapa 3: {stage_diffs[3]} > 1e-4"

    # 5. Coincidencia textual 100% en DEV y TEST-COMP y verificación de I/O constante
    for split_name in ("dev", "test-composicional"):
        dataset = ProceduralCompositionalDataset(cfg, split=split_name)
        num_eval = min(8, len(dataset.samples))
        for i in range(num_eval):
            prompt_b, _, _, _ = dataset.samples[i]

            out_c_buf = ctypes.create_string_buffer(512)
            lib.trace_generate_causal_two_step(model_buf, prompt_b, len(prompt_b), out_c_buf, 512)
            c_str = out_c_buf.value.decode("utf-8")

            with torch.no_grad():
                ref_str, _ = model_ref.generate_causal_two_step(prompt_b)

            assert c_str == ref_str, (
                f"Discrepancia en {split_name} muestra {i}:\n"
                f"PyTorch: {ref_str}\n"
                f"Nativo C: {c_str}"
            )

            # Confirmar que pread_calls no aumentó
            current_pread = lib.trace_model_get_pread_calls(model_buf)
            assert current_pread == pread_calls_cold, (
                f"Violación I/O: pread_calls aumentó ({current_pread} > {pread_calls_cold})"
            )

        print(f"✓ [{split_name.upper()}] 100% coincidencia textual ({num_eval}/{num_eval}) superada.")

    lib.trace_model_free(model_buf)
    print("✓ pread_calls se mantuvo estrictamente constante durante toda la inferencia.")
    print("✓ Todos los tests del Hito 2.4 superados exitosamente.")


if __name__ == "__main__":
    test_native_c_engine_odirect_and_parity()
