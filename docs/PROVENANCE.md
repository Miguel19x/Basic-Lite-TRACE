# Basic-Lite TRACE — Inventario de Procedencia y Trazabilidad (PROVENANCE)

Este documento contiene el inventario exhaustivo de procedencia de cada archivo, módulo, kernel, dataset y artefacto del proyecto **Basic-Lite TRACE**, conforme a las directrices de la OMPI para el aseguramiento de la cadena de autoría y custodia previa al registro de propiedad intelectual.

---

## 1. Tabla General de Trazabilidad y Procedencia

| Componente / Módulo | Procedencia / Autor | Versión / Commit Base | Licencia o Permiso | Modificaciones Propias Realizadas | Estado Legal Interno |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`trace/serialization/paged_bitcos.py`** | Creación original (Miguel19x) | Commit inicial v5 (`f379cb4`) | Propiedad exclusiva de Miguel19x | Código 100% original. Serialización y alineación a 4 KB para kernel Linux. | Verificado / Propio |
| **`csrc/paged_gemm_*.c`, `csrc/paged_dispatch.*`, `csrc/paged_unpack.h`** *(evolución de `paged_gemm_kernel.c`)* | Creación original (Miguel19x) | Commit inicial v5 (`f379cb4`) $\to$ v1.3.0 (`25b6c18`) / Fase 3 | Propiedad exclusiva de Miguel19x | Código 100% original en C con dispatch dinámico escalar/AVX2 y OpenMP para inferencia directa sobre páginas ternarias. | Verificado / Propio |
| **`csrc/trace_loader.c`, `csrc/trace_causal.c`, `csrc/trace_infer.h`, `csrc/trace_math.h`** *(evolución de `trace_infer.c`)* | Creación original (Miguel19x) | v1.3.0 (`25b6c18`) / Fase 3 | Propiedad exclusiva de Miguel19x | Cargador `O_DIRECT` + `pread` alineado a 4096 bytes y ejecutor causal standalone en C. | Verificado / Propio |
| **`csrc/lm/*` (`trace_tokenizer.c`, `trace_lm_loader.c`, `trace_lm_kv.c`, `trace_lm_infer.c`, cabeceras)** | Creación original (Miguel19x) | Rama Fase 3 (`exp/fase3-language-mla`) | Propiedad exclusiva de Miguel19x | Motor nativo C autorregresivo de lenguaje con tokenizador BPE compacto, caché KV e inferencia MLA. | Verificado / Propio |
| **`trace/lm/*` (`model.py`, `mla.py`, `config.py`, `tokenizer.py`)** | Creación original / Adaptación conceptual MLA (Miguel19x) | Rama Fase 3 (`exp/fase3-language-mla`) | Propiedad exclusiva de Miguel19x | Implementación propia en PyTorch de `TRACELanguageModel`, atención latente MLA ternaria y códec BPE. | Verificado / Propio |
| **`export_paged.py`, `export_paged_lm.py`** | Creación original (Miguel19x) | v1.3.0 (`25b6c18`) / Fase 3 | Propiedad exclusiva de Miguel19x | Exportadores binarios desde checkpoints PyTorch hacia formato `Paged BitCOS v1`. | Verificado / Propio |
| **`trace/layers/abacus.py`** | Creación original (Miguel19x) | Commit v6 (`b105127`) | Propiedad exclusiva de Miguel19x | Formulación vectorizada original de rangos decimales invertidos para números ASCII. | Verificado / Propio |
| **`trace/models/causal.py`** | Creación original (Miguel19x) | Commit v6 (`b105127`) | Propiedad exclusiva de Miguel19x | Arquitectura causal desacoplada en 2 etapas con anclaje `detach()`. | Verificado / Propio |
| **`trace/modules/paged_moe.py`** | Creación original (Miguel19x) | Commit v5 (`f379cb4`) | Propiedad exclusiva de Miguel19x | Algoritmo original de ruteo consciente de E/S física NVMe y buddy routing. | Verificado / Propio |
| **`trace/core/context_manager.py`** | Creación original (Miguel19x) | Commit v5 (`f379cb4`) | Propiedad exclusiva de Miguel19x | Arquitectura original de memoria jerárquica en 3 niveles (RAM, latente, SQLite FTS5). | Verificado / Propio |
| **`trace/layers/bitlinear.py`** | Adaptación conceptual (Miguel19x) | Basado en fórmula pública de BitNet (Wang et al., 2024) | Implementación propia (sin código copiado de Microsoft) | Reescritura propia desde cero en PyTorch utilizando `weight.sign()` y STE. | Verificado / Propio |
| **`trace/layers/norm.py`** | Adaptación conceptual (Miguel19x) | Basado en Zhang & Sennrich (RMSNorm) | Implementación propia | RMSNorm estándar implementado en PyTorch con epsilon configurable. | Verificado / Propio |
| **`data.py`** | Creación original (Miguel19x) | Commit v5 (`e8895f0`) | Propiedad exclusiva de Miguel19x | Generador sintético determinista de 2880 combinaciones y contrato de ancho fijo. | Verificado / Propio |
| **`train.py`** | Creación original (Miguel19x) | Commit v6 (`29ff4ca`) | Propiedad exclusiva de Miguel19x | Pipeline de entrenamiento con currículo Staged Warmup, schedule QAT BitNet b1.58 y telemetría. | Verificado / Propio |
| **`eval.py`, `eval_5seeds_summary.py`, `eval_diagnostico_colab.py`, `eval_fase1_5semillas.py`** | Creación original (Miguel19x) | Commit v6 (`113ee26`) $\to$ v1.2.0 | Propiedad exclusiva de Miguel19x | Suite de evaluación confirmatoria, desglose composicional, calibración ECE y contrastes pareados. | Verificado / Propio |
| **`results/`** | Evidencia científica oficial (Miguel19x) | Fases 1, 2 y 3 (Nov 2026) | Propiedad exclusiva de Miguel19x | Artefactos JSON de evaluación confirmatoria pre-registrada y benchmarks reproducibles. | Verificado / Propio |
| **Historial experimental (`archive/`)** | Histórico experimental (Miguel19x) | Etiquetas `prereg-v1` y `phase2-v6-stable` | Propiedad exclusiva de Miguel19x | Scripts exploratorios, ablaciones y modelos históricos preservados inmutablemente en Git tags. | Verificado / Propio |
| **Versión fundacional (`legacy/`)** | Repositorio originario (Miguel19x) | Etiqueta `v0.1-archive` (14-15 Nov 2026) | Propiedad exclusiva de Miguel19x | Versión 0.1 completa (50 archivos fuente) preservada inmutablemente en Git tag. | Verificado / Propio |
| **PyTorch (`torch`)** | PyTorch Foundation / Meta | v2.0+ (instalado vía pip) | BSD-3-Clause | Ninguna (dependencia externa de runtime). | Aprobado / Terceros |
| **NumPy (`numpy`)** | NumPy Developers | v1.24+ (instalado vía pip) | BSD-3-Clause | Ninguna (dependencia externa de runtime). | Aprobado / Terceros |
| **SQLite3 (`sqlite3`)** | SQLite Development Team | Estándar de Python 3 / C | Dominio Público | Ninguna (dependencia de almacenamiento de sistema). | Aprobado / Terceros |
| **`pytest` y `ruff`** *(solo CI/dev)* | pytest-dev / Astral Software | Instalados en GitHub Actions CI | MIT License | Ninguna (herramientas externas de prueba y linting). | Aprobado / Terceros |

---

## 2. Declaración de No Inclusión de Código Ajeno No Atribuido

* Se certifica que no se ha copiado código privado o protegido por derechos de autor de terceros sin la debida licencia o permiso.
* Los fragmentos que adaptan formulaciones teóricas de la literatura pública (BitNet, RoPE, RMSNorm, MLA) han sido reimplementados de forma independiente por Miguel19x sin reutilización de código fuente no licenciado.
* Ningún componente externo ha sido marcado indebidamente como propio.
