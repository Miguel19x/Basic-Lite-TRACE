# Changelog — Basic-Lite TRACE

Todas las modificaciones, versiones y evoluciones arquitectónicas de **Basic-Lite TRACE** quedan registradas en este documento siguiendo el formato de control de versiones y trazabilidad técnica.

---

## [TRACE-1.3.0] — Motor Nativo C Standalone (`O_DIRECT` + AVX2) y Cierre de Fase 2 — 2026-11-22

### Añadido y Consolidado
* **Motor Nativo C TRACE (`csrc/`, evolucionado desde `paged_gemm_kernel.c` y `trace_infer.c`):**
  - Serialización y carga directa mediante `O_DIRECT` + `pread` con *page pool* alineado a 4096 bytes (`export_paged.py`, `csrc/trace_loader.c`, `csrc/trace_causal.c`).
  - Micro-kernel GEMM ternario con tiling $K=512$, dispatch dinámico escalar/AVX2 (`csrc/paged_dispatch.c`, `csrc/paged_gemm_avx2.c`, `csrc/paged_gemm_scalar.c`) y ejecución estricta en 2 hilos (`OMP_NUM_THREADS=2`).
  - Paridad numérica exacta verificada frente a PyTorch AddSub (0 discrepancias de predicción, 0 pesos FP32 en `BitLinear`) y speedup medido de **3.74×–3.93×** (~75.59 ms/muestra en Intel Core i3-1115G4).
* **Gobernanza y Blindaje Legal:**
  - Actualización de `LICENSE.txt` (requisito de notificación previa para PoC de 30 días, acotación de titularidad frente a modelos de terceros, cláusula de divisibilidad y ley aplicable/idioma prevalente) y publicación de traducción informativa en inglés (`LICENSE-EN.md`).
  - Incorporación de plantillas de gobernanza en `.github/` y `CODE_OF_CONDUCT.md`.

---

## [TRACE-1.2.0] — Convergencia QAT BitNet b1.58 y Calibración — 2026-11-22

### Añadido y Consolidado
* **Resultados Empíricos Consolidados de Fase 1 (Exploratorio, post-prereg-v1, §9.4):**
  - Registro formal del artefacto canónico `results/eval_fase1_bitnet_5semillas_results.json` evaluando los checkpoints finales al paso 6000 (`_last.pt`) en CPU:
    - Test-Composicional Full-EM: media **$90.29\%$** (DE $13.44\%$, mediana $95.21\%$, IC95 bootstrap $[78.04\%, 97.29\%]$, pico $98.96\%$ en Semilla 456).
    - Ganancia neta de **$+71.25$ puntos porcentuales** sobre la línea base de Fase 2b ($19.04\% \to 90.29\%$).
    - Exact Match Scratchpad Etapa 1 ($P_w, P_{ms}$): **$100.00\% \pm 0.00\%$** en las 5 semillas (480/480).
    - Train-200 Full-EM (sin reemplazo, semilla 42): media **$99.70\% \pm 0.27\%$**.
    - Exactitud en dígitos aislados: **$98.94\% \pm 1.58\%$**.
    - Calibración Expected Calibration Error (ECE): media global **$0.61\%$** y Etapa 2 **$1.22\%$**. Entre el **$95.85\%$** y el **$99.88\%$** de las predicciones se ubicaron en la banda de alta confianza ($\ge 0.85$) con una exactitud observada de **$98.03\%$ a $99.95\%$** (media agregada: $98.96\%$ de cobertura a $99.41\%$ de exactitud).
  - Registro complementario de trazabilidad en `results/eval_fase1_5semillas_results.json` evaluando los picos intermedios de DEV (media Test-Comp $89.58\%$).
* **Schedule QAT de 2 Etapas estilo BitNet b1.58:**
  - Rampa lineal de LR en 300 pasos hacia $lr=1.0\text{e-}3$ con decaimiento coseno a $1.5\text{e-}4$ en 6000 pasos.
  - Decaimiento de peso desacoplado en 3 grupos de parámetros en `configure_adamw_groups()`: `weight_decay=0.05` en *shadow weights* de `BitLinear` durante la primera mitad (pasos 1-3000) descendiendo a `0.0` en la segunda mitad (pasos 3001-6000), manteniendo normas y embeddings en `0.0`.
  - Telemetría de saturación de activaciones (`act_saturation`) y tasa de cambio de signo ternario (`code_flip_rate`) integrada en `train.py`.
  - Calibración desacoplada de Etapa 2 (`stage2_ece_pct` en dígitos 6:12) integrada en `eval.py`.
* **Diagnóstico Causal y Lección Metodológica:**
  - Refutación empírica de la hipótesis de techo representacional: El estancamiento histórico de W1.58A8 respondía principalmente a subajuste de optimización bajo el estimador STE con hiperparámetros estándar de FP32. Al adaptar el schedule QAT de BitNet b1.58, el rendimiento saltó del 19.04% al 90.29%, aunque persiste varianza residual en la Semilla 123 (66.46%).
* **Gobernanza y Roadmap:**
  - Certificación formal y cierre de la **Fase 1** (`COMPLETADA`) y apertura de la **Fase 2** (`ACTIVA`: Motor Nativo C TRACE en i3-1115G4).

---

## [TRACE-1.1.0] — Desbloqueo Aritmético y Causalidad Decimal — 2026-11-21

### ⚠️ Ruptura de Compatibilidad (Breaking Changes)
* **Incompatibilidad Estructural con Checkpoints v1.0.0-trace (`trace_column_checkpoint.pt`):**
  - La carga de checkpoints anteriores mediante `load_state_dict()` falla por discrepancia estructural en `TRACECausalModel`:
    1. **Cabezal de Lenguaje (`lm_head`):** Migración desde `BitLinear` (que contenía pesos ternarios cuantizados y norma `sub_ln.weight`) a `nn.Linear(d_model, vocab_size, bias=False)` en FP32 de rango dinámico completo.
    2. **Condicionamiento de Etapa (`stage_emb`):** Reemplazo de las incrustaciones de tiempo/difusión (`time_emb`) por una incrustación discreta de etapa de 4 estados (`stage_emb = nn.Embedding(4, d_model)` para $stage\_id \in \{0, 1, 2, 3\}$).
    3. **Máscara Unificada de Target:** Sustitución de la máscara causal por bloques (`build_block_causal_mask`, `block_size=16`) por la máscara causal de bloque unificado bidireccional (`build_target_unified_mask`).
  *(Nota: El desacoplamiento causal entre Etapa 1 y Etapa 2 con `detach()` se mantiene intacto respecto a v1.0.0-trace).*

### Añadido y Consolidado
* **Resultados Empíricos Fase 2b (Exploratorio, post-prereg-v1, §9.4):**
  - Registro formal del artefacto `results/eval_fase2b_unblock_5semillas_results.json`:
    - Evaluación multi-semilla (42, 123, 456, 789, 999) tras 3000 pasos en Colab GPU con Train-200 muestreado sin reemplazo (§1.2.4).
    - Test-Composicional Full-EM: media $19.04\%$ (DE $22.73\%$, mediana $8.75\%$, IC95 bootstrap $[4.04\%, 43.12\%]$, pico $58.13\%$ en Semilla 999).
    - Exact Match Scratchpad Etapa 1 ($P_w, P_{ms}$): media $98.21\%$ (pares conocidos de entrenamiento).
    - Veredicto formal bajo H-A1: **INCONCLUSO** (IC95 cruza umbrales de 10% y 30%; contraste pareado cruza 0).
* **Diagnóstico Causal y Refutaciones Empíricas:**
  - *Refutación de grokking puro:* Train-200 y Test-Comp escalan en paralelo en las 5 semillas. El cuello de botella es subajuste de optimización QAT (meseta del STE con $lr=3\text{e-}4$), no sobreajuste.
  - *Refutación de bloqueo por acarreos/préstamos:* La exactitud en watt es plana (~73%-76%) entre `no_carry`, `single_carry` y `double_carry`. En ms, `borrow` ($91.7\%$) es el subcampo mejor resuelto.
  - *Firma de los 201 fallos:* Concentración en errores de una sola columna (`watt_±10`: 54 casos; `ms_±1`: 58 casos) y subajuste en pares específicos (`boost x media`: 15.0%).
* **Gobernanza y Arquitectura:**
  - Creación de `docs/ROADMAP_TRACE.md`: Hoja de ruta oficial de 6 fases con puertas de salida estrictas y separación estratégica en dos carriles (Carril A: Motor C en i3-1115G4 para pesos abiertos 2-bit; Carril B: Pesos propios TRACE en Colab GPU).

---

## [TRACE-1.0.0] — Arquitectura TRACE Convergente — 2026-11-17

### Añadido y Consolidado
* **Bautismo y Adopción de la Arquitectura TRACE:**
  - Consolidación formal de la arquitectura **TRACE** (*Ternary Resident Architecture for Convergent Execution*), aunando la cuantización ternaria W1.58A8, la residencia selectiva paginada a 4096 bytes en Linux, el ruteo MoE consciente de latencia NVMe y el campo causal en 2 etapas con `detach()`.
  - Transición del proyecto desde su génesis fundacional en **Basic-Lite 0.1** hacia la versión definitiva **Basic-Lite TRACE**, unificando la totalidad de los commits y el historial de desarrollo en el repositorio canónico.
  - Publicación y despliegue del régimen de **Licencia de Investigación, Docencia y Evaluación** en [LICENSE.txt](LICENSE.txt).
* **Reorganización Modular y Limpieza de Código Legacy:**
  - Migración y renombrado del paquete modular raíz de Python de `trdm/` a `trace/` para unificar el espacio de nombres con la arquitectura canónica TRACE.
  - Consolidación de todos los registros de evaluación confirmatoria pre-registrada en el directorio `results/`.
  - Purgado de código experimental histórico y de la versión fundacional (`archive/` y `legacy/`) del árbol de trabajo principal, quedando íntegramente preservados bajo las etiquetas Git congeladas `v0.1-archive`, `prereg-v1` y `phase2-v6-stable`.

---

## [0.2.0-v6] — Fase 2 Estable — 2026-11-16
**Etiqueta Git:** `phase2-v6-stable`  
**Commit principal:** `c874140` / `113ee26`

### Añadido
* **Campo Causal Desacoplado en 2 Etapas:**
  - Implementación en `trdm/models/causal.py` (anteriormente `trdm_v6_causal.py`) separando el cálculo de subtotales multiplicativos ($P_w, P_{ms}$) y la agregación aditiva final ($watt, ms$).
  - Anclaje causal con corte de gradiente (`detach()`) para mitigar de raíz el sesgo de exposición (*exposure bias*).
* **Abacus Positional Embeddings:**
  - Implementación 100% vectorizada en `trdm/layers/abacus.py` de rangos posicionales decimales invertidos (unidades=0, decenas=1, centenas=2) para tokens numéricos ASCII.
* **Hierarchical Byte Patcher ($P=4$):**
  - Implementación en `trdm/layers/byte_patcher.py` que agrupa 4 bytes por parche para reducir la complejidad cuadrática de atención por un factor de $16\times$.
* **Currículo Staged Warmup:**
  - Mecanismo en `train.py` (anteriormente `train_v6.py`) con `sched_p = 1.0` en épocas tempranas hacia `0.70` en fases avanzadas.
* **Evaluación Confirmatoria Oficial de 5 Semillas:**
  - Resultados oficiales en `eval_confirmatorio_5semillas_results.json` (etiqueta histórica `v6_confirmatorio_5semillas_results.json`):
    - Full Exact Match: $5.12\% \pm 4.88\%$ (máximo de $13.33\%$, rompiendo la nulidad histórica).
    - Exact Match Scratchpad ($P_w, P_{ms}$): $81.88\% \pm 10.14\%$ (máximo de $95.00\%$).
    - Validez Sintáctica: $100.00\% \pm 0.00\%$.
    - Exactitud en dígitos aislados: $81.97\% \pm 3.90\%$.

---

## [0.2.0-v5] — Fase 1 Pre-registrada — 2026-11-16
**Etiqueta Git:** `prereg-v1`  
**Commit base:** `f379cb4` / `d0a2b8b`

### Añadido
* **Paged BitCOS v1:**
  - Serialización binaria alineada a 4096 bytes del kernel de Linux con lectura directa vía `mmap` (`trdm/serialization/paged_bitcos.py`).
  - Formatos `fast` (2 bits/peso con LUT) y `compact` (presencia + signo).
* **Kernel C Fusionado Multi-Página:**
  - Implementación en `paged_gemm_kernel.c` con vectorización SIMD (AVX2) y paralelismo OpenMP sobre páginas binarias directamente en memoria mapeada.
* **MoE Consciente de Memoria Física:**
  - Algoritmo de enrutamiento en `trdm/modules/paged_moe.py` con penalización de latencia:
    $$S(e) = U(e) - \lambda C_{\text{lectura}}(e) - \mu C_{\text{cómputo}}(e)$$
  - Implementación de *buddy routing* ante cuellos de botella de disco.
* **Context Manager de 3 Niveles:**
  - Jerarquía desacoplada en `trdm/core/context_manager.py` (RAM activa, latente comprimido y persistencia en SQLite).
* **Protocolo Pre-registrado Oficial:**
  - Especificación histórica de experimentos E0 a E6 (preservada en la etiqueta `prereg-v1`).
  - Ejecución confirmatoria inicial de 5 semillas en Fase 1 (`archive/experiments_v4_v5/e1_confirmatorio_5semillas_results.json`).

---

## [0.1.0 - 0.1.4] — Versiones Exploratorias (TRDM-v1 a TRDM-v4) — 2026-11-14 a 2026-11-15
* Exploración de cuantización ternaria pura W1.58A8 con Straight-Through Estimator (STE).
* Implementación preliminar de kernel C para operaciones suma/resta sin multiplicadores flotantes (`ternary_kernel.c`).
* Pruebas de difusión discreta por bloques (CDLM).
* Detección de fallas de sobreajuste de longitud fija y diseño del contrato sintáctico de ancho fijo.
