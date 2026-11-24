# Basic-Lite TRACE — Informe Técnico de Benchmark de Inferencia y Eficiencia (Fase 3)

**Autor e Investigador Titular:** Miguel19x  
**Entorno de Hardware Certificado:** Intel Core i3-1115G4 (2 núcleos físicos / 4 hilos, 3.00 GHz, AVX2/FMA), 16 GB RAM DDR4, SSD NVMe, Linux x86_64.  
**Concurrencia de Ejecución:** `--cpu_threads 2` (`OMP_NUM_THREADS=2`)  
**Modelo de Evaluación:** `TRACELMConfig.preset_25m()` (12 capas, $d_{\text{model}}=512$, $d_{ff}=1024$, $n_{\text{heads}}=8$, $d_c=128$, $d_h=64$, $d_{pe}=32$, $d_v=64$, vocabulario BPE 4096, tied embeddings, 1,548 páginas Paged BitCOS v1.1).

---

## 1. Delimitación Epistémica y Alcance Metodológico

> [!IMPORTANT]
> **Condición de Falsación y Alcance de Hardware:**  
> Este benchmark audita de manera estricta y reproducible el rendimiento físico del hardware de consumo objetivo (latencia Time To First Token —TTFT—, velocidad de decodificación token a token $S=1$, aceleración por Prefix Caching y huella de memoria RAM pico `VmHWM`) empleando los pesos iniciales no entrenados del preset canónico de 25M parámetros.  
> La evaluación de convergencia semántica, coherencia del lenguaje natural y exactitud aritmética en tareas de razonamiento se reserva para el **Hito 3.6** (entrenamiento pre-registrado en Google Colab con aceleración GPU) y el cierre de versión formal `v1.4.0-trace` en el **Hito 3.7**. El presente Hito 3.5 certifica la viabilidad operativa y la ventaja computacional del motor nativo C desacoplado sobre CPU x86_64 estándar.

---

## 2. Resultados Empíricos Consolidados

Datos extraídos directamente de la corrida reproducible en subprocesos independientes (`results/fase3_benchmark_results.json`):

### Tabla 1: TTFT y Throughput de Decodificación ($S=1$) por Longitud de Contexto

| Variante | Memoria RAM Pico (`VmHWM`) | TTFT ($L=256$) | Decodificación ($L=256$) | TTFT ($L=1024$) | Decodificación ($L=1024$) | TTFT ($L=4096$) | Decodificación ($L=4096$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`trace_c_int8`** | **82.07 MB** | 391.51 ms | **254.30 tok/s** | 4,345.63 ms | **92.46 tok/s** | 56,947.44 ms | **32.19 tok/s** |
| **`trace_c_fp32`** | 101.46 MB | 456.95 ms | 215.80 tok/s | 4,132.26 ms | 105.80 tok/s | 57,996.02 ms | 30.09 tok/s |
| `pytorch_trace` | 2,854.64 MB | 283.83 ms | 48.41 tok/s | 1,828.74 ms | 40.81 tok/s | 23,886.15 ms | 23.40 tok/s |
| `pytorch_mha` (Baseline) | 1,836.14 MB | 138.58 ms | 85.31 tok/s | 1,164.36 ms | 42.72 tok/s | 13,878.51 ms | 18.37 tok/s |

---

### Tabla 2: Aceleración de TTFT con Prefix Caching Real ($L=256$)

Línea Base Fría (0% de prefijo reutilizado): **462.84 ms**

| Porcentaje Reuso | Tokens Cacheados en RAM | Tokens Procesados en C | Latencia TTFT (ms) | Speedup Empírico |
| :---: | :---: | :---: | :---: | :---: |
| **50%** | 128 | 128 | 294.39 ms | **1.57×** |
| **75%** | 192 | 64 | 145.10 ms | **3.19×** |
| **90%** | 230 | 26 | 77.87 ms | **5.94×** |

---

### Tabla 3: Auditoría Analítica Dinámica de KV Cache (12 capas Preset 25M)

| Arquitectura KV | Bytes / Token | Factor de Compresión vs MHA | Huella RAM a $L=256$ | Huella RAM a $L=1024$ | Huella RAM a $L=4096$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TRACE MLA INT8** | **3,120 B/tok** | **15.75× menor** | **0.76 MB** | **3.05 MB** | **12.19 MB** |
| **TRACE MLA FP32** | 7,680 B/tok | **6.40× menor** | 1.88 MB | 7.50 MB | 30.00 MB |
| **Canónico MHA FP32** | 49,152 B/tok | 1.00× *(Baseline)* | 12.00 MB | 48.00 MB | 192.00 MB |

---

## 3. Conclusiones Técnicas Principales

1. **Aislamiento y Eficiencia de Memoria:**
   - El motor nativo C con KV INT8 (`trace_c_int8`) operó con un pico físico de **82.07 MB**, frente a los **2,854.64 MB** demandados por PyTorch, lo que representa una **reducción de $34.8\times$ en el consumo de memoria RAM**.
   - El chunked prefill en bloques de $C=128$ tokens impidió que las matrices temporales de atención (`scores` y `attn_probs`) saturaran la RAM a $L=4096$.

2. **Throughput de Decodificación Autorregresiva ($S=1$):**
   - En generación token a token sobre contextos típicos ($L=256$), el motor nativo C alcanzó **254.30 tokens/segundo**, superando ampliamente a `pytorch_trace` (48.41 tok/s, **speedup de $5.25\times$**) gracias al kernel AVX2 Add-Sub en registros y la absorción completa de matrices de proyección ($W^{UK}$ y $W^{UV}$).
   - A $L=4096$, la decodificación se mantuvo en **32.19 tokens/segundo** (1.75× superior al baseline canónico denso de PyTorch).

3. **Efectividad del Prefix Caching:**
   - La reutilización del 90% del contexto histórico redujo el TTFT de 462.84 ms a **77.87 ms** (**$5.94\times$ de aceleración**), garantizando paridad exacta de logits ($\Delta = 0.00$) con respecto a la computación completa.

---

## 4. Cuello de Botella Identificado: TTFT en Frío a Contexto Largo y Hoja de Ruta

> [!WARNING]
> **Cuello de Botella de Cómputo en Prefill en Frío:**  
> Si bien el motor nativo C supera categóricamente a PyTorch en decodificación ($5.25\times$ a $L=256$) y en huella física de RAM ($34.8\times$ menor), el **TTFT en frío a contexto largo es más lento en C que en PyTorch**:
> - A $L=1024$: **4.35 s** (C INT8) vs **1.83 s** (PyTorch TRACE) / **1.16 s** (PyTorch MHA).
> - A $L=4096$: **56.95 s** (C INT8) vs **23.89 s** (PyTorch TRACE) / **13.88 s** (PyTorch MHA).
>
> **Causa Técnica Raíz en `csrc/lm/trace_lm_infer.c`:**
> 1. **Ausencia de Tiling BLAS en Prefill:** Mientras PyTorch delega la atención de prefill a kernels densos multihilo altamente vectorizados (OpenBLAS/MKL con paralelización OpenMP a nivel de secuencia completa), el prefill en C computa el producto escalar de atención token a token mediante bucles anidados sobre $t \in [0, S)$ y $s \in [0, \text{total\_len})$.
> 2. **Descuantización Repetida en el Bucle Interno:** En el bucle de atención causal, `trace_lm_kv_get_c_kv()` descuantiza el vector latente $c_{kv}$ ($d_c = 128$) de INT8 a FP32 en cada paso $s$ para cada una de las 8 cabezas. A $L=4096$, esto dispara más de 800 millones de descuantizaciones escalares en memoria caché L1/L2.
>
> **Mitigación Actual y Hoja de Ruta:**
> - En flujos de diálogo y agentes interactivos, esta sobrecarga queda amortizada por el **Prefix Caching**, que reduce el TTFT a $77.87\text{ ms}$ ($5.94\times$ de aceleración al reusar el 90% del contexto).
> - Se registra formalmente como **mejora pendiente para el motor C de Fase 4**: implementación de un kernel matricial de prefill por bloques (*blocked GEMM attention*) con descuantización en bloque de $c_{kv}$ previa a la contracción con las cabezas de consulta.
