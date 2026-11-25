# Basic-Lite TRACE: Standalone C (AVX2 + O_DIRECT) W1.58A8 Inference Engine for Consumer CPUs

> **Language / Idioma:** **English** | [Español](README.es.md)

![Status](https://img.shields.io/badge/status-experimental%20research-orange)
![Platform](https://img.shields.io/badge/platform-Linux%20x86__64-blue)
![Precision](https://img.shields.io/badge/precision-W1.58A8-brightgreen)
![License](https://img.shields.io/badge/license-Source--Available-red)
[![Basic-Lite TRACE CI](https://github.com/Miguel19x/Basic-Lite-TRACE/actions/workflows/ci.yml/badge.svg)](https://github.com/Miguel19x/Basic-Lite-TRACE/actions/workflows/ci.yml)

**Basic-Lite TRACE** (*Ternary Resident Architecture for Convergent Execution*) is an ultra-low-precision (**W1.58A8**) neural architecture and standalone C inference engine (**AVX2/FMA + `O_DIRECT` + MLA INT8 KV**) created and led by **Miguel19x**. On a 2-core consumer laptop CPU (**Intel Core i3-1115G4**, 16 GB RAM, NVMe SSD, Linux x86_64, no GPU), the standalone C runtime executes a 100% ternary **47.2M bilingual (EN/ES) language model** (`45m-bi`, **31.02 MB** 4 KB-aligned binary) at **214–237 tokens/s** in **36.22 MB peak RAM**, with Python $\to$ C export parity of `max_diff` $= 3.34 \times 10^{-6}$ and zero FP32 weight expansion in RAM.

Architecturally, TRACE is governed by **selective residency and execution** (`Paged BitCOS`, 4 KB OS-page alignment, I/O-aware MoE routing, and 3-tier context memory), designed to scale toward multi-billion-parameter NVMe-paged models by decoupling total disk capacity from active working RAM under pre-registered, reproducible protocols.

---

## 1. Empirical Results & Hardware Benchmarks

### 1.1. Phase 3 — Bilingual `45m-bi` W1.58 LM & Standalone C Engine (v1.4.0-trace, Milestone 3.7)

**Milestone 3.7** completes Phase 3 with a 100% ternary W1.58 bilingual (EN/ES) autoregressive language model (**`45m-bi`**, **47.2M parameters**) pretrained on Kaggle (`2× NVIDIA Tesla T4`, DDP, 7h 43m 03s) for 8,000 steps (**524.3M tokens**) via Continual QAT (FP16W warmup steps 1–1200 $\to$ strict W1.58 STE steps 1201–8000), converging to $\text{Train Loss} = 2.88$, $\text{Val Loss} = 3.09$, **$\text{Val PPL} = 22.08$** (full reports: [docs/HITO_3_7_BENCHMARKS.md](docs/HITO_3_7_BENCHMARKS.md), [docs/FASE3_BENCHMARK.md](docs/FASE3_BENCHMARK.md)).

- **Model Specs:** $L=14$, $d_{\text{model}}=640$, $d_{\text{ff}}=1280$ (Squared-ReLU), $n_{\text{heads}}=10$, MLA $d_c=160$, $d_h=64$, $d_{pe}=32$, $V=8,192$ BPE tied embeddings.
- **Physical Binary (`Paged BitCOS v1.1`):** **31.02 MB** (32,522,240 bytes), 2,716 4 KB pages aligned for `O_DIRECT`, export parity `max_diff` $= 3.34 \times 10^{-6}$.

![Milestone 3.7 Training Curve](docs/assets/hito37_training_curve_en.svg)

#### Standalone CPU Performance (`Intel Core i3-1115G4`, 2 physical cores, `OMP_NUM_THREADS=2`)
Measured with `/usr/bin/time -v ./trace_lm` over 100 generated tokens:

| ISA Dispatch | KV Cache Mode | Greedy TTFT (ms) | Greedy Throughput (tok/s) | Sampling Throughput (tok/s) | Peak RAM (Max RSS) | Speedup vs Scalar |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AVX2 + FMA** | **INT8 KV** (default) | **16.94 ± 2.17 ms** | **214.53 ± 27.31 tok/s** | **162.81 tok/s** | **36.22 MB** | **8.11× / 20.49× TTFT** |
| **AVX2 + FMA** | **FP32 KV** (`--no_int8_kv`) | **18.17 ± 1.90 ms** | **237.26 ± 10.78 tok/s** | **171.48 tok/s** | **36.66 MB** | **9.76× / 20.79× TTFT** |
| **Scalar C99** | **INT8 KV** | 347.09 ± 69.78 ms | 26.47 ± 3.48 tok/s | 20.89 tok/s | 36.22 MB | Scalar INT8 Baseline |
| **Scalar C99** | **FP32 KV** | 377.73 ± 74.55 ms | 24.30 ± 0.17 tok/s | 24.91 tok/s | 36.89 MB | Scalar FP32 Baseline |

### 1.2. Phase 1 — Compositional Arithmetic Convergence (v1.2.0-trace, 5 Fixed Seeds)
Evaluated across 5 fixed seeds (`42, 123, 456, 789, 999`) at step 6000 on CPU (2 threads, i3-1115G4), logged in [results/eval_fase1_bitnet_5semillas_results.json](results/eval_fase1_bitnet_5semillas_results.json):

| Metric | Phase 2b (Baseline) | **Phase 1 (BitNet QAT Schedule, v1.2.0)** | Delta |
| :--- | :---: | :---: | :---: |
| **Test-Comp Full Exact Match (Full-EM)** | $19.04\% \pm 22.73\%$ (Median: $8.75\%$) | **$90.29\% \pm 13.44\%$** (Median: **$95.21\%$**) | **+71.25 pp** |
| **Scratchpad EM ($P_w, P_{ms}$)** | $98.21\% \pm 1.01\%$ | **$100.00\% \pm 0.00\%$** (480/480 across 5 seeds) | **+1.79 pp** |
| **Isolated Digit Accuracy** | $88.20\% \pm 5.61\%$ | **$98.94\% \pm 1.58\%$** | **+10.74 pp** |
| **Train-200 Full-EM (without replacement)** | $24.90\% \pm 26.23\%$ | **$99.70\% \pm 0.27\%$** | **+74.80 pp** |
| **DEV Full-EM (108 cases)** | $21.67\% \pm 25.64\%$ | **$94.81\% \pm 10.07\%$** | **+73.14 pp** |
| **Test-Random Full-EM (240 cases)** | $19.58\% \pm 22.84\%$ | **$94.58\% \pm 8.18\%$** | **+75.00 pp** |
| **Test-Comp ECE Calibration** | *Not logged* | **$0.61\%$** (Stage 2: $1.22\%$) | **Well-calibrated** |

```text
[Seed  42] Test-Comp: EM = 94.17% (452/480) | SP-EM = 100.00% | ECE = 0.37% | DEV: EM =  98.15%
[Seed 123] Test-Comp: EM = 66.46% (319/480) | SP-EM = 100.00% | ECE = 2.01% | DEV: EM =  76.85%
[Seed 456] Test-Comp: EM = 98.96% (475/480) | SP-EM = 100.00% | ECE = 0.06% | DEV: EM = 100.00%
[Seed 789] Test-Comp: EM = 95.21% (457/480) | SP-EM = 100.00% | ECE = 0.35% | DEV: EM = 100.00%
[Seed 999] Test-Comp: EM = 96.67% (464/480) | SP-EM = 100.00% | ECE = 0.25% | DEV: EM =  99.07%
------------------------------------------------------------------------------------------------
Global Mean  : Test-Comp: EM = 90.29% ± 13.44% (95% Bootstrap CI: [78.04%, 97.29%])
Scratchpad EM: Test-Comp: EM = 100.00% ± 0.00% (480/480 across all 5 seeds)
```

*(Historical pre-registered v1.0.0 baseline: Test-Comp Full-EM $5.12\% \pm 4.88\%$, Scratchpad EM $81.88\% \pm 10.14\%$; see [results/eval_confirmatorio_5semillas_results.json](results/eval_confirmatorio_5semillas_results.json)).*

---

## 2. Quickstart & Reproduction

```bash
# 1. Create virtual environment & install package/dependencies
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Build standalone native C engine & kernels (GCC + OpenMP + AVX2, -Wall -Wextra -Werror -Wvla)
OMP_NUM_THREADS=2 make clean && make all

# 3. Run verification tests
OMP_NUM_THREADS=2 make test

# 4. Run standalone C LM inference (requires exported Paged BitCOS v1.1 .bin + BPE tokenizer)
./trace_lm --prompt "Once upon a time, a curious engineer" --temp 0.0 --max_tokens 30
./trace_lm --prompt "El algoritmo de inferencia profunda" --temp 0.7 --top_p 0.9 --rep_pen 1.15 --max_tokens 35

# 5. Local CPU pilot training for Two-Stage Causal Field (2 physical threads)
python train.py --steps 1000 --batch_size 32 --warmup_ratio 0.40 --device cpu
```

> [!NOTE]
> Binary checkpoints (`*.pt`, `*.bin`) are excluded via `.gitignore`. Checkpoints can be generated locally or tracked per release in [GitHub Releases](https://github.com/Miguel19x/Basic-Lite-TRACE/releases) (latest published release: `v1.3.0-trace`; `v1.0.0-trace` artifacts are historical and structurally incompatible with `v1.2.0+`).

---

## 3. Architecture Overview & Five Canonical Axes

```mermaid
flowchart TD
    subgraph InputStage ["1. Input & Positional Encoding"]
        RawBytes["Raw Text / Input Stream"]:::input --> BytePatch["ASCII Bytes (260, Ph. 1–2) / Compact BPE (4K–8K, Ph. 3, 1:1 Digits)"]:::input
        RawBytes --> AbacusLayer["Abacus Positional Embedding (Ph. 1–2: U=0, D=1, C=2)"]:::input
        BytePatch & AbacusLayer --> CombinedEmbed["Token & Positional Representations"]:::input
    end

    subgraph CoreTernary ["2. Resident Ternary Core (W1.58A8 — AVX2/FMA C Kernels)"]
        CombinedEmbed --> SharedCore["BitLinear W1.58A8 Core (0 FP32 Weights in RAM)"]:::core
        SharedCore --> MLA["Phase 3: Ternary MLA (4x–15.75x KV) + Squared-ReLU FFN"]:::core
        SharedCore --> PagedMoE["Phases 1–2: Ternary GQA (4:2) + Resident MoE (4 Experts + Shared)"]:::core
    end

    subgraph TwoStageField ["3. Execution Heads & Output Contracts"]
        PagedMoE --> Stage1["Ph. 1–2 STAGE 1: Subtotals (pw, pms)"]:::causal
        Stage1 --> CausalAnchor["Causal Anchor with Stop-Gradient (detach)"]:::gate
        CausalAnchor --> Stage2["Ph. 1–2 STAGE 2: Conditioned Totals (watt, ms)"]:::causal
        Stage2 --> OutputBytes["Fixed-Width Output (watt: 000 / ms: 000)"]:::causal
        MLA --> LMHead["Phase 3: Tied Embeddings LM Head (trace_lm, 214–237 tok/s)"]:::lm
    end

    subgraph MemoryHierarchy ["4. Physical Resource Managers"]
        PagedBitCOS[("Paged BitCOS v1/v1.1 (4 KB Pages on NVMe)")]:::storage <--> PagePool["Contiguous RAM Page Pool (O_DIRECT + pread)"]:::runtime
        PagePool <--> PagedMoE
        L1RAM["Tier 1: Active RAM (INT8 / FP32 KV)"]:::runtime <--> L2Latent["Tier 2: Compressed Latents"]:::runtime <--> L3SQLite[("Tier 3: SQLite FTS5 on Disk")]:::storage
    end

    classDef input fill:#0c4a6e,stroke:#38bdf8,stroke-width:2px,color:#f0f9ff
    classDef core fill:#3b0764,stroke:#a78bfa,stroke-width:2px,color:#f5f3ff
    classDef causal fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#ecfdf5
    classDef gate fill:#78350f,stroke:#fbbf24,stroke-width:2px,color:#fef3c7
    classDef lm fill:#881337,stroke:#fb7185,stroke-width:2px,color:#ffe4e6
    classDef runtime fill:#1e293b,stroke:#94a3b8,stroke-width:2px,color:#f1f5f9
    classDef storage fill:#7c2d12,stroke:#fb923c,stroke-width:2px,color:#ffedd5

    style InputStage fill:#082f49,stroke:#38bdf8,stroke-width:1.5px,color:#e0f2fe
    style CoreTernary fill:#2e1065,stroke:#a78bfa,stroke-width:1.5px,color:#ede9fe
    style TwoStageField fill:#022c22,stroke:#34d399,stroke-width:1.5px,color:#d1fae5
    style MemoryHierarchy fill:#431407,stroke:#fb923c,stroke-width:1.5px,color:#ffedd5
```

### Core Pillars & Five Axes
1. **OS-Page-Aligned Binary Paging (`Paged BitCOS v1 / v1.1`):** 100% strict ternary weights $\{-1, 0, 1\}$ ([trace/layers/bitlinear.py](trace/layers/bitlinear.py)) executed directly from **4096-byte aligned pages** via `O_DIRECT` + `pread` and AVX2/FMA Add-Sub C kernels ([csrc/paged_dispatch.c](csrc/paged_dispatch.c), [csrc/paged_gemm_avx2.c](csrc/paged_gemm_avx2.c), [trace/serialization/paged_bitcos.py](trace/serialization/paged_bitcos.py)).
2. **I/O-Aware Paged MoE Routing:** Penalizes non-resident experts via $S(e) = U(e) - \lambda C_{\text{read}}(e) - \mu C_{\text{compute}}(e)$ with *buddy expert* fallback under strict latency budgets ([trace/modules/paged_moe.py](trace/modules/paged_moe.py)).
3. **Multi-Head Latent Attention (MLA) & 3-Tier Context Memory:** Joint KV latent compression ([trace/lm/mla.py](trace/lm/mla.py), **4×–8×** FP32, **15.75×** INT8 KV) paired with Active RAM, Compressed Latents, and SQLite FTS5 persistent storage ([trace/core/context_manager.py](trace/core/context_manager.py)).
4. **Compact Byte/BPE Tokenization:** Evolved from raw ASCII bytes (260 tokens, Phase 1–2) to a native C compact BPE tokenizer ([csrc/lm/trace_tokenizer.c](csrc/lm/trace_tokenizer.c), [trace/lm/tokenizer.py](trace/lm/tokenizer.py), 4K–8K vocabulary, 1:1 digit preservation, tied embeddings) in Phase 3.
5. **Abacus Embeddings & Two-Stage Causal Field:** Vectorized right-to-left decimal rank embeddings ([trace/layers/abacus.py](trace/layers/abacus.py)) + stop-gradient (`detach()`) causal anchoring ([trace/models/causal.py](trace/models/causal.py)) + Staged Warmup curriculum ([train.py](train.py)).

### Architecture Decision Record (ADR)

| Architectural Decision | Technical Rationale | Validated Advantage | Trade-Off |
| :--- | :--- | :--- | :--- |
| **Paged BitCOS v1 (4 KB)** | Direct `O_DIRECT` execution without inflating layers to FP32. | 1.58-bit weights; OS-page-aligned disk reads. | Requires matrix packing aligned to 4 KB multiples. |
| **I/O-Aware MoE Router** | Prevent PCIe/NVMe saturation from indiscriminate expert swaps. | Reduces physical disk transfers by favoring RAM-resident experts. | Requires tuning $\lambda_{\text{read}}$ and *buddy expert* fallbacks. |
| **Multi-Head Latent Attention** | Shrink active KV cache retained in working RAM. | 4×–15.75× KV RAM reduction (FP32 / INT8). | Adds linear decompression projection at inference. |
| **Abacus Embeddings** | Give ASCII numeric tokens explicit decimal place-value symmetry. | Boosts isolated digit accuracy from 60.5% to >81.9% (98.94% in v1.2). | Requires vectorized reverse cumulative rank indexing. |
| **Two-Stage Causal Field** | Eliminate Exact Match collapse under parallel factorization. | Scales Test-Comp Full-EM from 0% to 90.29% ± 13.44% (peak 98.96%). | Requires two forward passes at inference. |
| **Staged Warmup Curriculum** | Prevent noisy Stage 1 outputs from corrupting Stage 2 addition. | Stable convergence under scheduled sampling. | Requires calibrating warmup ratio ($40\%$ of steps). |

---

## 4. Repository Structure

```
.
├── .github/                                  # CI (Ubuntu + ASan/UBSan) & GitHub governance templates
├── csrc/                                     # Modular native C engine (O_DIRECT, AVX2/FMA, OpenMP)
│   ├── lm/                                   # Autoregressive LM C engine (BPE, MLA, INT8 KV, trace_lm)
│   ├── paged_dispatch.{c,h}                  # Thread-safe scalar / AVX2 dispatch for ternary GEMM
│   ├── paged_gemm_avx2.c / _scalar.c         # AVX2 Add-Sub & portable C99 GEMM micro-kernels
│   ├── trace_loader.c / trace_causal.c       # 4 KB O_DIRECT loader & 2-stage causal CLI (trace_infer)
│   └── trace_infer.h / trace_math.h / paged_unpack.h
├── trace/                                    # Canonical Python package
│   ├── core/                                 # 3-tier context manager (RAM/Latent/SQLite FTS5) & types
│   ├── layers/                               # Abacus, BitLinear W1.58A8, FFN, RMSNorm, RoPE, Embeddings
│   ├── lm/                                   # TRACELanguageModel, ternary MLA, config & BPE tokenizer
│   ├── modules/                              # Attention, PagedMoE & SharedResidentTernaryCore
│   ├── models/                               # TRACECausalModel (Two-Stage Causal Field)
│   ├── ops/                                  # ctypes bindings to libpaged_gemm.so & CPU ternary ops
│   └── serialization/                        # Paged BitCOS v1 / v1.1 reader/writer
├── scripts/ / notebooks/                     # LM corpus, DDP training, benchmarks & GPU pilot notebooks
├── tests/                                    # 15 verification suites (AVX2, O_DIRECT, parity, MLA, E0, CLI)
├── docs/                                     # Benchmarks, roadmaps, provenance, SHA-256 + RFC 3161 (.tsr)
├── results/                                  # Confirmatory multi-seed JSONs & benchmark artifacts
├── pyproject.toml / requirements.txt / Makefile / ruff.toml
├── LICENSE.txt / LICENSE-EN.md               # Binding Source-Available license (ES) & reference (EN)
├── CONTRIBUTING.md / CODE_OF_CONDUCT.md / SECURITY.md / NOTICE.md / AUTHORS.md / CHANGELOG.md
└── THIRD_PARTY_NOTICES.md / licenses/        # Third-party open-source attributions (BSD-3, MIT, PD)
```

---

## 5. Collaboration Model, Licensing & Citation

### Collaboration Model (SQLite-Style 3-Channel Funnel)
Basic-Lite TRACE is published under a **Source-Available** license to enable rigorous scientific scrutiny, teaching, and legitimate technical evaluation while reserving commercial exploitation. Following an **SQLite-style governance model**, the core repository does not merge unsolicited external pull requests without a prior bilateral agreement, while offering **three active channels** (see [CONTRIBUTING.md](CONTRIBUTING.md)):

* **Community Hardware Benchmarks (Issues):** Run `make all` / `./trace_lm` on other CPUs (AMD Ryzen, Intel AVX-512, ARM64 / Apple Silicon) and share reproducible latency and RAM metrics.
* **Bug Reports & Technical Diagnostics (Issues):** Report defects or profile bottlenecks (e.g., cold-prefill optimization in `csrc/lm/trace_lm_infer.c`).
* **Core Team (Bilateral Agreement):** Looking for long-term collaborators in **C / SIMD / Linux Systems** and **ML Scaling (PyTorch)**. The best way to start is by opening a high-signal technical Issue or writing to `miguelaranguren19x@outlook.com`.

### Ownership, IP & Governance
* **Sole Author & Rights Holder:** Miguel19x (`miguelaranguren19x@outlook.com`)
* **License:** [LICENSE.txt](LICENSE.txt) (binding Spanish Source-Available license; research, teaching, and 30-day PoC with prior written notice; commercial/SaaS and commercial AI training/distillation reserved) · English reference translation: [LICENSE-EN.md](LICENSE-EN.md)
* **Governance & Policies:** [NOTICE.md](NOTICE.md) · [CONTRIBUTING.md](CONTRIBUTING.md) · [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) · [SECURITY.md](SECURITY.md) · [AUTHORS.md](AUTHORS.md) · [CHANGELOG.md](CHANGELOG.md) · [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
* **Provenance & Cryptographic Integrity:** [docs/PROVENANCE.md](docs/PROVENANCE.md) · [docs/RELEASE_MANIFEST.md](docs/RELEASE_MANIFEST.md) (external RFC 3161 timestamp: `docs/RELEASE_MANIFEST.md.tsr`) · [docs/AI_ASSISTANCE.md](docs/AI_ASSISTANCE.md)

### Citation & Commercial Licensing
```bibtex
@misc{miguel19x2026basiclitetrace,
  author = {Miguel19x},
  title = {Basic-Lite TRACE: Ternary Resident Architecture for Convergent Execution on Consumer CPUs},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Miguel19x/Basic-Lite-TRACE}}
}
```

For **commercial licensing**, **30-day PoC start notifications**, or **custom C/SIMD deployment consulting**, contact **Miguel19x** at `miguelaranguren19x@outlook.com`.
