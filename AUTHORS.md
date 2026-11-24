# Basic-Lite TRACE — Registro de Autores, Contribuciones y Créditos

Este documento detalla la autoría, el diseño arquitectónico y las atribuciones técnicas del proyecto **Basic-Lite TRACE** (arquitectura **TRACE**).

---

## 1. Fundador y Responsable Técnico

* **Autor y Director del Proyecto:** **Miguel19x**
  * **Contacto:** `miguelaranguren19x@outlook.com`
  * **Función:** Creador único e inventor de la arquitectura. Concibió la idea unificada de integrar todo este modelo y sistema en una arquitectura coherente, dirigiendo la evolución completa desde la fundación inicial en **Basic-Lite 0.1** (familias TRDM-v1 a v4) hasta la consolidación definitiva en **Basic-Lite TRACE** (arquitectura **TRACE**). Es el responsable de la formulación matemática de los mecanismos, la programación de módulos y kernels, el diseño del protocolo experimental y la dirección técnica de la investigación.

---

## 2. Desglose de Contribuciones Técnicas

### 2.1. Concepción Arquitectónica y Diseño
* **Responsable:** Miguel19x
* **Aportaciones clave:**
  - Creación y formulación de la arquitectura **TRACE** (*Ternary Resident Architecture for Convergent Execution*, nacida del linaje TRDM), concibiendo la unificación del cómputo ternario W1.58A8, la residencia selectiva paginada, la recurrencia dinámica en estados latentes y el campo causal desacoplado en 2 etapas.
  - Formulación de la hipótesis de **residencia y ejecución selectivas** (desacoplar la capacidad total en disco de 30B de la memoria RAM activa en procesadores de consumo básico).
  - Formulación matemática del algoritmo de **enrutamiento MoE con penalización por latencia física de E/S de almacenamiento** ($S(e) = U(e) - \lambda C_{\text{lectura}} - \mu C_{\text{cómputo}}$) y contingencia por *buddy routing*.
  - Diseño de la arquitectura de **Campo Causal en 2 Etapas** con anclaje de subtotales mediante stop-gradient (`detach()`), eliminando el sesgo de exposición (*exposure bias*).
  - Especificación del formato de serialización binaria **Paged BitCOS v1** con alineación nativa a 4096 bytes del kernel de Linux.
  - Jerarquía de compresión de contexto en tres niveles (RAM activa, compresión latente, persistencia SQLite).
  - Integración de *Abacus Positional Embeddings* con empaquetamiento jerárquico de bytes ($P=4$).

### 2.2. Programación e Implementación
* **Responsable:** Miguel19x
* **Aportaciones de código:**
  - Módulos del paquete canónico `trace/` (capas, atención MLA, ruteadores MoE paginados, núcleo residente compartido `trace/modules/resident_core.py`, modelo causal `trace/models/causal.py`).
  - Kernels nativos en C optimizados con SIMD/OpenMP (`paged_gemm_kernel.c` y variantes fundacionales preservadas en las etiquetas Git `v0.1-archive` y `phase2-v6-stable`).
  - Pipelines de entrenamiento (`train.py` con currículo *Staged Warmup*; versiones históricas preservadas en etiquetas Git).
  - Generador determinista de datos sintéticos (`data.py`) y motor de evaluación de 2 etapas (`eval.py`), con artefactos de evidencia consolidados en `results/`.

### 2.3. Metodología de Evaluación y Control de Sesgos
* **Marco de trabajo:** Metodología experimental determinista con separación estricta entre entrenamiento, validación y prueba composicional.
* **Higiene E0:** Verificación empírica de causalidad temporal en bloques, aislamiento estricto de muestras de lote y consistencia de parsers numéricos.
* **Evaluación confirmatoria:** Corridas independientes con 5 semillas fijadas (`42, 123, 456, 789, 999`) sobre 480 casos de prueba composicional.

---

## 3. Créditos de Software de Código Abierto y Literatura Científica

El proyecto se apoya con gratitud en las siguientes bibliotecas de código abierto y formulaciones del estado del arte:

* **Bibliotecas base:**
  - **PyTorch:** Framework de tensores y autograd (PyTorch Foundation / Meta AI / The Linux Foundation).
  - **NumPy:** Computación matricial y utilidades de evaluación (NumPy Developers).
  - **SQLite:** Motor de base de datos de dominio público integrado en la biblioteca estándar de Python.
* **Referencias conceptuales:**
  - **BitNet b1.58 (Wang et al., Microsoft Research):** Concepto de cuantización ternaria $\{-1, 0, 1\}$ con Straight-Through Estimator (STE).
  - **Multi-Head Latent Attention (MLA, DeepSeek-AI):** Concepto de compresión de claves y valores (KV) en un espacio latente conjunto.
  - **RoPE (Su et al.):** Incrustación posicional rotatoria.
  - **RMSNorm (Zhang & Sennrich):** Normalización de raíz cuadrada media.
  - **Abacus Embeddings (McLeish et al., 2024):** Formulación de incrustaciones de dígitos con alineación decimal recursiva («Transformers Can Do Arithmetic with the Right Embeddings»).
