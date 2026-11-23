"""
Basic-Lite TRACE — Benchmark Riguroso de Rendimiento y Memoria de Fase 3
Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
Todos los derechos reservados. Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

Mide:
- TTFT (Time To First Token) en frío para L in {256, 1024, 4096}.
- Throughput de decodificación autorregresiva (tokens/s) S=1.
- Aceleración por Prefix Caching (0%, 50%, 75%, 90% reuso).
- Huella de memoria física pico (VmHWM / RSS) en subprocesos aislados.
- Desglose analítico de bytes/token del KV cache (MLA INT8 vs MLA FP32 vs MHA FP32).

NOTA METODOLÓGICA:
Este benchmark audita exclusivamente características de hardware (velocidad, latencia
y memoria RAM) empleando pesos no entrenados del preset TRACELMConfig.preset_25m().
La evaluación semántica y aritmética se reserva para el Hito 3.6 (entrenamiento en Google Colab)
y el cierre formal v1.4.0-trace para el Hito 3.7.
"""

import argparse
import ctypes
import json
import math
import os
import resource
import subprocess
import sys
import tempfile
import time

LIB_PATH = os.path.abspath("libtrace_lm.so")
RESULTS_PATH = os.path.abspath("results/fase3_benchmark_results.json")


def get_peak_rss_mb() -> float:
    """Extrae la memoria física pico (VmHWM) del subproceso actual sin dependencias."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    parts = line.split()
                    return float(parts[1]) / 1024.0
    except Exception:
        pass
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


# =====================================================================
# Funciones Auxiliares de Ejecución en C (con Chunked Prefill Seguro)
# =====================================================================
def load_c_lib():
    lib = ctypes.CDLL(LIB_PATH)
    lib.trace_lm_load.argtypes = [ctypes.c_char_p]
    lib.trace_lm_load.restype = ctypes.c_void_p
    lib.trace_lm_free.argtypes = [ctypes.c_void_p]
    lib.trace_lm_workspace_create.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.trace_lm_workspace_create.restype = ctypes.c_void_p
    lib.trace_lm_workspace_free.argtypes = [ctypes.c_void_p]
    lib.trace_lm_kv_create.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.trace_lm_kv_create.restype = ctypes.c_void_p
    lib.trace_lm_kv_free.argtypes = [ctypes.c_void_p]
    lib.trace_lm_kv_reset.argtypes = [ctypes.c_void_p]
    lib.trace_lm_forward.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.trace_lm_forward.restype = ctypes.c_int
    lib.trace_lm_sample_greedy.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.trace_lm_sample_greedy.restype = ctypes.c_int
    lib.trace_lm_kv_get_last_common_prefix.argtypes = [ctypes.c_void_p]
    lib.trace_lm_kv_get_last_common_prefix.restype = ctypes.c_int
    return lib


def run_c_chunked_prefill(lib, c_model, c_kv, c_ws, prompt: list[int], chunk_size: int = 128):
    """Ejecuta prefill en bloques de chunk_size tokens pasando siempre last_token_only=1."""
    L = len(prompt)
    vocab_size = 4096
    c_logits = (ctypes.c_float * vocab_size)()

    for chunk_start in range(0, L, chunk_size):
        chunk_end = min(chunk_start + chunk_size, L)
        sub_tokens = prompt[:chunk_end]
        c_tokens = (ctypes.c_int * chunk_end)(*sub_tokens)
        # Siempre last_token_only = 1 para evitar buffer overflow en c_logits
        lib.trace_lm_forward(c_model, c_kv, c_tokens, chunk_end, c_logits, 1, c_ws)
    return c_logits


# =====================================================================
# Definición de Clases PyTorch (Importadas Solo Bajo Demanda)
# =====================================================================
def get_canonical_mha_class():
    from trace.layers.norm import RMSNorm
    from trace.lm.model import TRACELMConfig

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class CanonicalMHACausalLayer(nn.Module):
        def __init__(self, d_model: int, n_heads: int, d_ff: int):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.attn_norm = RMSNorm(d_model)
            self.ffn_norm = RMSNorm(d_model)
            self.ffn_up = nn.Linear(d_model, d_ff, bias=False)
            self.ffn_down = nn.Linear(d_ff, d_model, bias=False)

        def forward(self, x: torch.Tensor, k_cache=None, v_cache=None):
            B, S, D = x.shape
            h = self.attn_norm(x)
            q = self.q_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

            if k_cache is not None and v_cache is not None:
                k = torch.cat([k_cache, k], dim=2)
                v = torch.cat([v_cache, v], dim=2)

            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            total_len = k.shape[2]
            if S > 1:
                mask = torch.triu(
                    torch.full((S, total_len), float("-inf"), device=x.device), diagonal=total_len - S + 1
                )
                scores = scores + mask
            probs = torch.softmax(scores, dim=-1)
            out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, S, D)
            x = x + self.out_proj(out)
            ffn_h = F.relu(self.ffn_up(self.ffn_norm(x))) ** 2
            x = x + self.ffn_down(ffn_h)
            return x, k, v

    class CanonicalDenseMHALM(nn.Module):
        def __init__(self, cfg: TRACELMConfig):
            super().__init__()
            self.cfg = cfg
            self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.layers = nn.ModuleList(
                [CanonicalMHACausalLayer(cfg.d_model, cfg.n_heads, cfg.d_ff) for _ in range(cfg.n_layers)]
            )
            self.final_norm = RMSNorm(cfg.d_model)

        def forward(self, x: torch.Tensor, kv_cache=None):
            B, S = x.shape
            h = self.tok_emb(x)
            new_kvs = []
            for i, layer in enumerate(self.layers):
                k_c, v_c = kv_cache[i] if kv_cache else (None, None)
                h, k_out, v_out = layer(h, k_c, v_c)
                new_kvs.append((k_out, v_out))
            h = self.final_norm(h)
            logits = F.linear(h, self.tok_emb.weight)
            return logits, new_kvs

    return CanonicalDenseMHALM


# =====================================================================
# Ejecutor Aislado por Subproceso (Worker)
# =====================================================================
def run_worker(variant: str, model_bin: str, context_lens: list[int], decode_steps: int):
    results = {}
    vocab_size = 4096

    if variant in ["trace_c_int8", "trace_c_fp32"]:
        # Subproceso C puro: SIN import de torch para RSS limpio
        lib = load_c_lib()
        c_model = lib.trace_lm_load(model_bin.encode())
        use_int8 = 1 if variant == "trace_c_int8" else 0
        c_ws = lib.trace_lm_workspace_create(c_model, 128)

        for L in context_lens:
            prompt = [(i * 31 + 5) % vocab_size for i in range(L)]
            c_kv = lib.trace_lm_kv_create(c_model, L + decode_steps + 16, use_int8)

            # 1. Medir TTFT (Chunked Prefill)
            t0 = time.perf_counter()
            logits_last = run_c_chunked_prefill(lib, c_model, c_kv, c_ws, prompt, chunk_size=128)
            ttft_ms = (time.perf_counter() - t0) * 1000.0

            # 2. Medir Throughput de Decodificación (S = 1)
            next_tok = lib.trace_lm_sample_greedy(logits_last, vocab_size)
            curr_prompt = list(prompt)
            t_dec_start = time.perf_counter()

            for _ in range(decode_steps):
                curr_prompt.append(next_tok)
                c_tokens = (ctypes.c_int * len(curr_prompt))(*curr_prompt)
                c_out = (ctypes.c_float * vocab_size)()
                lib.trace_lm_forward(c_model, c_kv, c_tokens, len(curr_prompt), c_out, 1, c_ws)
                next_tok = lib.trace_lm_sample_greedy(c_out, vocab_size)

            t_dec_total = time.perf_counter() - t_dec_start
            tok_per_sec = decode_steps / max(1e-6, t_dec_total)

            results[str(L)] = {
                "ttft_ms": round(ttft_ms, 2),
                "decode_tok_per_sec": round(tok_per_sec, 2),
            }
            lib.trace_lm_kv_free(c_kv)

        lib.trace_lm_workspace_free(c_ws)
        lib.trace_lm_free(c_model)

    elif variant == "pytorch_trace":
        from trace.lm.model import TRACELanguageModel, TRACELMConfig

        import torch

        torch.set_num_threads(2)
        cfg = TRACELMConfig.preset_25m()
        cfg.max_seq_len = 4096 + decode_steps + 16
        model = TRACELanguageModel(cfg)
        model.eval()

        for L in context_lens:
            prompt = [(i * 31 + 5) % vocab_size for i in range(L)]
            inp = torch.tensor([prompt], dtype=torch.long)

            with torch.no_grad():
                # TTFT
                t0 = time.perf_counter()
                logits, _, kv_caches = model(inp, use_weight_absorption=True, quantize_kv_int8=False)
                ttft_ms = (time.perf_counter() - t0) * 1000.0

                # Decode S=1
                next_tok = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                t_dec_start = time.perf_counter()
                curr_tok = torch.tensor([[next_tok]], dtype=torch.long)

                for _ in range(decode_steps):
                    dec_logits, _, kv_caches = model(
                        curr_tok,
                        kv_caches=kv_caches,
                        use_weight_absorption=True,
                        quantize_kv_int8=False,
                    )
                    next_tok = int(torch.argmax(dec_logits[:, -1, :], dim=-1).item())
                    curr_tok = torch.tensor([[next_tok]], dtype=torch.long)

                t_dec_total = time.perf_counter() - t_dec_start
                tok_per_sec = decode_steps / max(1e-6, t_dec_total)

            results[str(L)] = {
                "ttft_ms": round(ttft_ms, 2),
                "decode_tok_per_sec": round(tok_per_sec, 2),
            }

    elif variant == "pytorch_mha":
        from trace.lm.model import TRACELMConfig

        import torch

        torch.set_num_threads(2)
        cfg = TRACELMConfig.preset_25m()
        cfg.max_seq_len = 4096 + decode_steps + 16
        DenseMHAClass = get_canonical_mha_class()
        model = DenseMHAClass(cfg)
        model.eval()

        for L in context_lens:
            prompt = [(i * 31 + 5) % vocab_size for i in range(L)]
            inp = torch.tensor([prompt], dtype=torch.long)

            with torch.no_grad():
                # TTFT
                t0 = time.perf_counter()
                logits, kv_cache = model(inp)
                ttft_ms = (time.perf_counter() - t0) * 1000.0

                # Decode S=1
                next_tok = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                t_dec_start = time.perf_counter()
                curr_tok = torch.tensor([[next_tok]], dtype=torch.long)

                for _ in range(decode_steps):
                    dec_logits, kv_cache = model(curr_tok, kv_cache=kv_cache)
                    next_tok = int(torch.argmax(dec_logits[:, -1, :], dim=-1).item())
                    curr_tok = torch.tensor([[next_tok]], dtype=torch.long)

                t_dec_total = time.perf_counter() - t_dec_start
                tok_per_sec = decode_steps / max(1e-6, t_dec_total)

            results[str(L)] = {
                "ttft_ms": round(ttft_ms, 2),
                "decode_tok_per_sec": round(tok_per_sec, 2),
            }

    peak_rss = get_peak_rss_mb()
    worker_payload = {
        "variant": variant,
        "peak_rss_mb": round(peak_rss, 2),
        "results": results,
    }
    print(json.dumps(worker_payload))


# =====================================================================
# Benchmark de Prefix Caching
# =====================================================================
def run_prefix_caching_benchmark(model_bin: str) -> dict:
    lib = load_c_lib()
    c_model = lib.trace_lm_load(model_bin.encode())
    c_ws = lib.trace_lm_workspace_create(c_model, 128)
    vocab_size = 4096
    L = 256

    prompt_a = [(i * 47 + 13) % vocab_size for i in range(L)]

    # 1. Frío (0% reutilizado)
    c_kv_cold = lib.trace_lm_kv_create(c_model, 512, 1)
    t0 = time.perf_counter()
    run_c_chunked_prefill(lib, c_model, c_kv_cold, c_ws, prompt_a, chunk_size=128)
    cold_ms = (time.perf_counter() - t0) * 1000.0
    lib.trace_lm_kv_free(c_kv_cold)

    reuse_cases = {
        "50%": int(L * 0.50),  # 128 tokens
        "75%": int(L * 0.75),  # 192 tokens
        "90%": int(L * 0.90),  # 230 tokens
    }

    pfx_data = {"cold_ttft_ms": round(cold_ms, 2), "cases": {}}

    for label, reuse_len in reuse_cases.items():
        # Reiniciar y precargar prompt_a antes de cada caso para evitar contaminación cruzada
        c_kv = lib.trace_lm_kv_create(c_model, 512, 1)
        run_c_chunked_prefill(lib, c_model, c_kv, c_ws, prompt_a, chunk_size=128)

        # Construir prompt con reuse_len tokens idénticos y el resto nuevos
        prompt_var = prompt_a[:reuse_len] + [((i + 999) * 17) % vocab_size for i in range(L - reuse_len)]
        c_tokens_var = (ctypes.c_int * len(prompt_var))(*prompt_var)
        c_out = (ctypes.c_float * vocab_size)()

        t_start = time.perf_counter()
        lib.trace_lm_forward(c_model, c_kv, c_tokens_var, len(prompt_var), c_out, 1, c_ws)
        cached_ms = (time.perf_counter() - t_start) * 1000.0

        pfx_recorded = lib.trace_lm_kv_get_last_common_prefix(c_kv)
        assert pfx_recorded == reuse_len, f"Prefix caching discrepancia: {pfx_recorded} != {reuse_len}"

        speedup = cold_ms / max(1e-4, cached_ms)
        pfx_data["cases"][label] = {
            "cached_tokens": reuse_len,
            "processed_tokens": L - reuse_len,
            "ttft_ms": round(cached_ms, 2),
            "speedup": round(speedup, 2),
        }
        lib.trace_lm_kv_free(c_kv)

    lib.trace_lm_workspace_free(c_ws)
    lib.trace_lm_free(c_model)
    return pfx_data


# =====================================================================
# Orquestador Principal
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Basic-Lite TRACE Benchmark Fase 3")
    parser.add_argument("--worker", type=str, default=None, help="Ejecutar variante en subproceso aislado")
    parser.add_argument("--model_bin", type=str, default=None, help="Ruta al binario de modelo exportado")
    parser.add_argument("--output", type=str, default=RESULTS_PATH, help="Ruta de salida JSON")
    args = parser.parse_args()

    context_lens = [256, 1024, 4096]
    decode_steps = 32

    # Si es modo worker, ejecutar y salir inmediatamente
    if args.worker:
        run_worker(args.worker, args.model_bin, context_lens, decode_steps)
        return

    # Modo Orquestador Principal
    print("=" * 80)
    print("Basic-Lite TRACE — Benchmark Formal de Inferencia y Eficiencia (Fase 3)")
    print("Objetivo de Hardware: Intel Core i3-1115G4 (2 hilos físicos) | Preset 25M")
    print("=" * 80)

    # 1. Exportar modelo de prueba en directorio temporal con max_seq_len seguro
    with tempfile.TemporaryDirectory() as tmp_dir:
        from trace.lm.model import TRACELanguageModel, TRACELMConfig
        from trace.serialization.paged_bitcos import PagedBitCOSSerializer

        tmp_model_bin = os.path.join(tmp_dir, "benchmark_25m.bin")
        print("\n[1/5] Exportando TRACELMConfig.preset_25m() a Paged BitCOS v1.1...")
        cfg = TRACELMConfig.preset_25m()
        cfg.max_seq_len = 4096 + decode_steps + 16
        model = TRACELanguageModel(cfg)
        model.eval()
        PagedBitCOSSerializer.export_paged_model(model=model, output_path=tmp_model_bin, format_type="fast")

        # 2. Ejecutar cada variante en subproceso aislado para medición limpia de RSS
        variants = ["trace_c_int8", "trace_c_fp32", "pytorch_trace", "pytorch_mha"]
        bench_data = {"hardware": "Intel Core i3-1115G4 (2 threads)", "variants": {}}

        print("[2/5] Ejecutando benchmarks aislados en subprocesos independientes...")
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "2"
        env["PYTHONPATH"] = "."

        for var in variants:
            print(f"  -> Ejecutando variante: {var}...")
            cmd = [
                sys.executable,
                __file__,
                "--worker",
                var,
                "--model_bin",
                tmp_model_bin,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            var_json = json.loads(res.stdout.strip())
            bench_data["variants"][var] = var_json

        # 3. Benchmark de Prefix Caching
        print("[3/5] Evaluando Prefix Caching en C (reuso 0%, 50%, 75%, 90%)...")
        pfx_bench = run_prefix_caching_benchmark(tmp_model_bin)
        bench_data["prefix_caching"] = pfx_bench

        # 4. Cálculo analítico dinámico de KV Cache para las 12 capas reales de preset_25m()
        print("[4/5] Computando desglose analítico de huella KV Cache para 12 capas...")
        mla_i8_b_tok = cfg.n_layers * (cfg.kv_lora_rank * 1 + 4 + cfg.qk_rope_head_dim * 4)
        mla_fp32_b_tok = cfg.n_layers * (cfg.kv_lora_rank * 4 + cfg.qk_rope_head_dim * 4)
        mha_fp32_b_tok = cfg.n_layers * (cfg.n_heads * (cfg.d_model // cfg.n_heads) * 4 * 2)

        kv_analytics = {
            "bytes_per_token": {
                "TRACE_MLA_INT8": mla_i8_b_tok,
                "TRACE_MLA_FP32": mla_fp32_b_tok,
                "CANONICAL_MHA_FP32": mha_fp32_b_tok,
            },
            "compression_vs_mha": {
                "TRACE_MLA_INT8": round(mha_fp32_b_tok / mla_i8_b_tok, 2),
                "TRACE_MLA_FP32": round(mha_fp32_b_tok / mla_fp32_b_tok, 2),
            },
            "total_kv_mb_at_context": {
                str(ctx): {
                    "MLA_INT8": round((mla_i8_b_tok * ctx) / (1024 * 1024), 2),
                    "MLA_FP32": round((mla_fp32_b_tok * ctx) / (1024 * 1024), 2),
                    "MHA_FP32": round((mha_fp32_b_tok * ctx) / (1024 * 1024), 2),
                }
                for ctx in context_lens
            },
        }
        bench_data["kv_cache_analytics"] = kv_analytics

        # 5. Guardar JSON estructurado
        print(f"[5/5] Guardando resultados en {args.output}...")
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(bench_data, f, indent=2)

    # Renderizar Tablas Markdown en Consola
    print("\n" + "=" * 80)
    print("TABLA 1: TTFT y Rendimiento de Decodificación por Longitud de Contexto")
    print("=" * 80)
    print(
        f"{'Variante':<22} | {'RSS Pico':<10} | {'TTFT L=256':<11} | {'Dec L=256':<10} | {'TTFT L=1024':<11} | {'Dec L=1024':<10} | {'TTFT L=4096':<11} | {'Dec L=4096':<10}"
    )
    print("-" * 115)
    for var, vdata in bench_data["variants"].items():
        rss = f"{vdata['peak_rss_mb']} MB"
        r = vdata["results"]
        t256 = f"{r['256']['ttft_ms']} ms"
        d256 = f"{r['256']['decode_tok_per_sec']} t/s"
        t1024 = f"{r['1024']['ttft_ms']} ms"
        d1024 = f"{r['1024']['decode_tok_per_sec']} t/s"
        t4096 = f"{r['4096']['ttft_ms']} ms"
        d4096 = f"{r['4096']['decode_tok_per_sec']} t/s"
        print(
            f"{var:<22} | {rss:<10} | {t256:<11} | {d256:<10} | {t1024:<11} | {d1024:<10} | {t4096:<11} | {d4096:<10}"
        )

    print("\n" + "=" * 80)
    print("TABLA 2: Aceleración de TTFT con Prefix Caching (Contexto L=256)")
    print("=" * 80)
    cold_t = bench_data["prefix_caching"]["cold_ttft_ms"]
    print(f"Línea Base Fría (0% reutilizado): {cold_t} ms")
    print(f"{'Reuso (%)':<12} | {'Tokens Cacheados':<16} | {'Tokens Procesados':<18} | {'TTFT (ms)':<10} | {'Speedup':<8}")
    print("-" * 75)
    for label, cdata in bench_data["prefix_caching"]["cases"].items():
        print(
            f"{label:<12} | {cdata['cached_tokens']:<16} | {cdata['processed_tokens']:<18} | {cdata['ttft_ms']:<10} | {cdata['speedup']:<8.2f}x"
        )

    print("\n" + "=" * 80)
    print("TABLA 3: Auditoría Analítica Dinámica de KV Cache (12 capas Preset 25M)")
    print("=" * 80)
    print(f"{'Arquitectura KV':<20} | {'Bytes / Token':<14} | {'Compresión vs MHA':<18} | {'L=4096 RAM':<12}")
    print("-" * 72)
    i8_tok_str = f"{mla_i8_b_tok:,} B/tok"
    i8_cmp_str = f"{kv_analytics['compression_vs_mha']['TRACE_MLA_INT8']}x menor"
    i8_mem_str = f"{kv_analytics['total_kv_mb_at_context']['4096']['MLA_INT8']} MB"

    fp32_tok_str = f"{mla_fp32_b_tok:,} B/tok"
    fp32_cmp_str = f"{kv_analytics['compression_vs_mha']['TRACE_MLA_FP32']}x menor"
    fp32_mem_str = f"{kv_analytics['total_kv_mb_at_context']['4096']['MLA_FP32']} MB"

    mha_tok_str = f"{mha_fp32_b_tok:,} B/tok"
    mha_cmp_str = "1.00x (Baseline)"
    mha_mem_str = f"{kv_analytics['total_kv_mb_at_context']['4096']['MHA_FP32']} MB"

    print(f"{'TRACE MLA INT8':<20} | {i8_tok_str:<14} | {i8_cmp_str:<18} | {i8_mem_str:<12}")
    print(f"{'TRACE MLA FP32':<20} | {fp32_tok_str:<14} | {fp32_cmp_str:<18} | {fp32_mem_str:<12}")
    print(f"{'Canónico MHA FP32':<20} | {mha_tok_str:<14} | {mha_cmp_str:<18} | {mha_mem_str:<12}")
    print("=" * 80)
    print("✓ Benchmark formal completado exitosamente.")


if __name__ == "__main__":
    main()
