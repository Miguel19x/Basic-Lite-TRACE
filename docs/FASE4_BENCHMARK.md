# Basic-Lite TRACE — Benchmarks Consolidados de Fase 4 (4A + 4B)

**Hardware de referencia:** Intel Core i3-1115G4 (Tiger Lake, 2 núcleos físicos / 4 hilos lógicos, AVX-512 VNNI + AVX2 + FMA + F16C), 16 GB DDR4-3200, SSD NVMe PCIe Gen4 x4, Linux x86_64, sin GPU.  
**Ejecución estricta:** `OMP_NUM_THREADS=2`, 0 expansión de pesos `BitLinear` a FP32 en RAM.  
**Fecha:** 25 de noviembre de 2026.

---

## 1. Fase 4A — Motor Nativo C a Escala 2B Real (`microsoft/bitnet-b1.58-2B-4T`)

Los pesos abiertos de terceros `microsoft/bitnet-b1.58-2B-4T` (licencia MIT; véase [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) §3) fueron convertidos en streaming puro Python (`tools/import_bitnet.py`) a formato **`Paged BitCOS v1.1`** (GQA 20 Q / 5 KV, `ReLU²` gated FFN, `sub_ln`, embeddings `FP16`, `1.2 GB` en disco) + tokenizador **`TRTK v2`** (128,256 tokens LLaMA-3 Byte-Level BPE, `4.3 MB`).

### 1.1. Rendimiento Cabeza a Cabeza vs. `bitnet.cpp` (2 núcleos físicos, `OMP_NUM_THREADS=2`)

| Métrica | **Basic-Lite TRACE (AVX-512 VNNI + `O_DIRECT`)** | `bitnet.cpp` (GGUF oficial sin parchear) | `bitnet.cpp` (parcheado `--override-kv llama-bpe`) |
| :--- | :---: | :---: | :---: |
| **Prefill `pp128`** | **109.24 – 113.69 tok/s** (`1.13–1.17 s`) | — | — |
| **Prefill `pp512`** | **83.12 tok/s** (`6.16 s`) | 60.4 – 80.8 tok/s | 60.4 – 80.8 tok/s |
| **Decode `tg32` (`S=1`)** | **12.50 – 15.01 tok/s** | 14.18 – 15.17 tok/s | 14.18 – 15.17 tok/s |
| **WikiText-2 PPL (`2×512`)** | **16.24** | 54.09 | 30.46 |
| **RAM Pico (Max RSS)** | **1,255.3 MB** | ~1.3 GB | ~1.3 GB |
| **Carga física (`O_DIRECT`)** | **422.9 ms** (`2 pread_calls`) | — | — |

### 1.2. Optimizaciones Clave de Fase 4A
* **Prefill en frío (`462d520`):** Kernel MLA de 4 cabezas, descuantización KV en una sola pasada, paralelización OpenMP y reducción de ~30 MB en `ws->scores` (4.9×–7.4× aceleración de TTFT en preset 25M).
* **GQA + Embeddings Compactos (`8b5fec0`):** Atención GQA 4:1, RoPE split-half con `rope_theta` configurable, FFN gated `ReLU²`, `sub_ln` opcional y embeddings `FP16`/`INT8` en `Paged BitCOS v1.1` (evita 1.31 GB de expansión FP32).
* **Tokenizador `TRTK v2` 128K (`71f2c59`):** Tabla `byte_to_id[256]`, `TraceMergeV2`, plantilla de chat LLaMA-3 (`--chat` / `--system`) y parada en `eot_id`.
* **Kernel AVX-512 VNNI (`fe2ae81`):** `csrc/paged_gemm_avx512.c` usando `vpdpbusd` sin resta de 1 ($\sum w_{\text{u8}} x - \sum x$), micro-kernels `4×4` y `2×4` ZMM, `K_TILE_SIZE = 8192` en una sola pasada, cuantización/descuantización INT8 vectorizada AVX-512/AVX2 y `lm_head` FP16 de 8 filas.

---

## 2. Fase 4B — Banco de 4 Dominios + Matriz `C1–C3` + Capa TIR + Sandbox

Artefacto JSON completo: [results/fase4b_bench_2b_c1_c3.json](../results/fase4b_bench_2b_c1_c3.json) (`--max-new-tokens 48`, `OMP_NUM_THREADS=2`, tiempo total `152.8 s` sobre `checkpoints/bitnet_2b_fp16.bin`).

| Condición | Overall Full-EM | Código / Terminal | Mates / Lógica | Docs Largos (`needle` / `synthesis`) | Factual | Response Brier ↓ | JSON TIR válidos | Escapes de sandbox |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`C1`** (Modelo solo) | **25.0%** (`2/8`) | 0.0% | 0.0% | **100.0%** (`1.0` / `1.0`) | 0.0% | `0.3482` | — | `0` |
| **`C2`** (+ SQLite L3) | **50.0%** (`4/8`) | 0.0% | 0.0% | **100.0%** (`1.0` / `1.0`) | **100.0%** | **`0.2365`** | — | `0` |
| **`C3`** (+ L3 + TIR) | **37.5%** (`3/8`) | 0.0% | 0.0% | 50.0% (`1.0` / `0.0`) | **100.0%** | `0.3476` | `2 / 8` | **`0`** |

### Conclusiones Empíricas de Fase 4B
1. **Seguridad:** `out_of_sandbox_executions == 0` en toda la corrida y 17/17 tests adversariales de jaula/AST/L3 en verde (`tests/test_tir_sandbox.py`).
2. **Memoria L3 offline (`C2` vs. `C1`):** SQLite FTS5 + BM25 eleva `factual` de `0.0%` a `100.0%` y mejora `response_brier` de `0.3482` a `0.2365` sin conexión a red.
3. **Brecha base 2B en `C3` (motivación de Fase 5):** La máscara PDA JSON en C (`csrc/lm/trace_grammar.c`) garantiza prefijos 100% válidos, pero el modelo 2B base sin ajuste fino genera objetos verbosos que se truncan dentro del presupuesto de 24 tokens de llamada TIR y no invoca herramientas de forma calibrada. Esta brecha es el objetivo directo de **Fase 5 (`<think>` + TIR SFT + GRPO corto)**.
