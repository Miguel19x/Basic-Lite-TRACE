# Plan Técnico y Protocolo Experimental Post-Fase 5 — Basic-Lite TRACE

**Titular e Inventor:** Miguel19x  
**Contacto Oficial:** `miguelaranguren19x@outlook.com`  
**Licencia:** Source-Available ([LICENSE.txt](../LICENSE.txt))  
**Estado:** `BORRADOR DE DISEÑO / CONGELADO`  
**Dependencia obligatoria:** Cierre verificado de Fase 2 (`v1.3.0-trace`), Fase 3, Fase 4 y Fase 5 según `docs/ROADMAP_TRACE.md`.  
**Hardware objetivo de referencia:** CPU Intel Core i3-1115G4 (2 núcleos físicos / `2 hilos`, AVX2, 16 GB DDR4, SSD NVMe, sin GPU dedicada).

---

## 1. Propósito y Principio Rector

El objetivo de largo plazo de Basic-Lite TRACE tras cerrar la Fase 5 es competir en hardware de consumo doméstico en **tres dimensiones simultáneas**:
1. **Contexto largo utilizable sin colapso de precisión** (tanto en localización puntual como en síntesis multi-pasaje).
2. **Alta relación calidad / costo por token** en tres dominios verificables: **programación y herramientas locales**, **razonamiento lógico-matemático** y **análisis exacto de documentos largos**.
3. **Entrenamiento y adaptación accesibles, modulares y auditables** con presupuesto doméstico o GPU modesta.

Para evitar promesas no demostradas, todo el diseño se rige por la **Regla de Separación de 4 Carriles**:
- **Velocidad del ejecutor** (C / AVX2 / `O_DIRECT` / `io_uring`) $\neq$ **Capacidad del modelo** (arquitectura y pesos).
- **Capacidad del modelo** $\neq$ **Calidad del sistema** (memoria L3, recuperación, herramientas y verificadores).
- **Inferencia barata** (1.58–2 bits) $\neq$ **Entrenamiento barato** (FLOPs, *rollouts*, datos y ensayos fallidos).

Ninguna fase avanza asumiendo que las reducciones teóricas de memoria o cómputo se suman automáticamente en tiempo de pared ni que una técnica funciona sin medición directa en el hardware objetivo.

---

## 2. Decisiones Arquitectónicas Fijadas (Puente Fases 2 $\to$ 5)

Antes de entrar a la Fase 6, el repositorio consolida estas bases operativas:

| Fase | Alcance consolidado | Regla técnica fijada |
| :--- | :--- | :--- |
| **Fase 2 (Activa)** | Motor nativo C (`trace_infer.c`), `Paged BitCOS v1`, AVX2, `O_DIRECT` + *page pool* | Mantener **bytes 1:1** (vocab 260). Puerta: paridad $\le 1\text{e-}4$, 3 falsaciones, 0 FP32 en `BitLinear`, speedup $\ge 3\times$ en 2 hilos. |
| **Fase 3** | `TRACELanguageModel` (15M–50M) en C | Línea base de lenguaje con **BPE compacto (4K–8K tokens, embeddings compartidos, dígitos 1:1)** + **MLA compacto** + **Prefix Caching y KV cache en INT8** en `trace_infer.c`. |
| **Fase 4** | Pesos abiertos 2-bit en Motor C | Soporte BPE nativo en C y paridad de perplejidad $\pm 1\%$ frente a la referencia. Despliegue inicial del banco de evaluación antes de entrenar especialistas. |
| **Fase 5** | MoE $>16$ GB paginado en NVMe + MTP + SQLite L3 | Bloques contiguos por experto alineados a 4096 B, lectura directa asíncrona (`O_DIRECT` / `io_uring`), caché selectiva **LRU/LFU** en RAM, cabezales MTP / borrador especulativo y memoria externa L3 en SQLite. |

---

## 3. Plan Detallado Post-Fase 5 (Fases 6 a 9)

### [FASE 6 — CONGELADA] Banco de Evaluación de 3 Dominios y Sistema Base

**Objetivo:** Medir con rigor de dónde proviene realmente la precisión (pesos vs. memoria externa vs. herramientas vs. adaptación) antes de invertir GPU en entrenar especialistas.

#### 3.1. Diseño de los 3 conjuntos de evaluación (con tests ocultos)
1. **Programación y herramientas locales:**
   - Tareas de edición de código acotada, depuración y uso de utilidades locales en bucle cerrado (editar $\to$ compilar/ejecutar $\to$ leer error $\to$ corregir).
   - Evaluación estricta con **tests ocultos** separados de cualquier suite usada durante entrenamiento o verificación interna.
2. **Razonamiento lógico-matemático:**
   - Problemas nuevos fuera de distribución que distingan explícitamente entre **planteamiento correcto** y **cálculo exacto** (con y sin evaluador simbólico/numérico).
3. **Documentos largos:**
   - Dividido obligatoriamente en dos subpruebas:
     - *Localización puntual:* encontrar cifras, fechas, cláusulas o citas exactas.
     - *Razonamiento documental multi-pasaje:* combinar evidencia repartida en varias secciones, resolviendo versiones, excepciones y contradicciones.

#### 3.2. Matriz experimental (4 configuraciones × comparación simétrica)
Sobre **un único modelo base sin alterar su arquitectura**, se evalúan 4 configuraciones bajo un **presupuesto explícito por respuesta** (tiempo máximo, RAM pico, número de intentos y límite de llamadas a herramientas):

| Config. | Descripción | Qué aísla |
| :--- | :--- | :--- |
| **C1** | Modelo solo | Capacidad intrínseca de los pesos base |
| **C2** | Modelo + Recuperación documental (SQLite L3) | Aporte real de la memoria externa y calidad del *retriever* |
| **C3** | Modelo + L3 + Herramientas y verificadores | Aporte del bucle cerrado de ejecución y comprobación |
| **C4** | Modelo adaptado/especializado + mismas herramientas | Aporte neto del entrenamiento adicional frente a C3 |

- **Regla de comparación justa:** Cuando se compare contra un modelo mayor (referencia externa), el modelo mayor **también debe evaluarse con las mismas herramientas, memoria y presupuesto** (C2/C3). Superar a un modelo grande *sin* herramientas solo valida el diseño del sistema, no mayor capacidad del modelo pequeño.
- **Auditoría de calibración (ECE y Puntuación Brier):** Medir por separado:
  1. Calibración a nivel de token vía ECE.
  2. Calibración de **corrección factual a nivel de respuesta completa mediante puntuación Brier sobre aciertos en tests ocultos** (separada de la ECE de logits/tokens). No se asume que la calibración de dígitos de Fase 1 (`0.61%` ECE) proteja contra alucinaciones en texto abierto; cualquier umbral para disparar consultas a L3 debe calibrarse y probarse en este banco.

**Puerta de salida de Fase 6:**
- Banco reproducible automatizado para los 3 dominios con tests ocultos.
- Tabla comparativa C1–C3 vs. modelo mayor bajo el mismo presupuesto, identificando fallos concretos y cuantificados antes de abrir Fase 8.

---

### [FASE 7 — CONGELADA] Escalamiento de Contexto Largo sin Degradación de Precisión

**Objetivo:** Descubrir el techo real de contexto útil en 16 GB de RAM + NVMe mediante incrementos aislados y medidos, sin asumir viabilidad de 1M de tokens a priori.

#### 3.3. Escalera experimental obligatoria (1 variable por paso)

| Escalón | Cambio aislado | Qué valida |
| :--- | :--- | :--- |
| **7.1 Referencia** | MLA compacto + Prefix Caching + KV INT8 en contexto moderado (4K–16K) | Línea base de RAM total, tiempo de ingestión, ms/token y exactitud documental |
| **7.2 Híbrido modular** | Intercalar capas de estado constante $O(1)$ (lineal/SSM, p. ej. proporción inicial configurable **3:1**) con capas de atención comprimida | Ahorro real de RAM frente a pérdida de recuerdo exacto; medir además el coste real de entrenamiento/adaptación continua de los pesos híbridos.<br>**Nota crítica de diseño en Prefix Caching:** A diferencia del KV-cache de atención tradicional (que permite cortes e indexación directa por token), un estado recurrente/SSM $O(1)$ no puede recortarse arbitrariamente. El Prefix Caching sobre capas recurrentes/SSM requiere **checkpointing de estado por bloques (*chunks*) contiguos** para posibilitar bifurcaciones o reutilización de prefijos sin recomputación desde cero. |
| **7.3 Atención dispersa (*Sparse*)** | Limitar el presupuesto de bloques consultados por token en las capas de atención | Separar presupuesto de **almacenamiento** (cuánto se guarda) vs. **lectura** (cuánto se escanea); medir ganancia en latencia sin perder recuperación lejana |
| **7.4 Persistencia y TTT opcional** | Guardado/restauración incremental en disco (estado $O(1)$ checkpointed por *chunks* + bloques latentes + índices) + prueba aislada de **Test-Time Training (TTT)** ligero sobre un subconjunto de pesos MLP | Evitar recálculo de prefijos largos y evaluar si unos pocos pasos de adaptación en tiempo de prueba mejoran la síntesis en contextos largos frente a L3 + *sparse*. Medir explícitamente el impacto en latencia de prefill (TTFT) en CPU i3. |

#### 3.4. Presupuesto real de memoria y métricas obligatorias
En cada salto de longitud (4K $\to$ 16K $\to$ 64K $\to$ 256K $\to$ techo físico en 16 GB), se reportan **siempre** cinco cifras:
1. **RAM pico real del proceso** (contabilizando latentes, componentes posicionales RoPE, escalas de cuantización, índices de bloques, alineación y buffers de trabajo —nunca solo el mínimo teórico de latentes).
2. **Tiempo total de ingestión (*prefill*)** y de restauración desde disco.
3. **Latencia sostenida por token** (mediana y p95 en 2 hilos).
4. **Exactitud de localización** (aguja puntual) vs. **exactitud de síntesis** (multi-pasaje con excepciones).
5. **Tasa de lectura real de NVMe** si se habilita paginación de bloques fríos.

**Puerta de salida de Fase 7:**
- Curva empírica completa de contexto vs. RAM, latencia y exactitud en los dos subconjuntos documentales, demostrando restauración exacta de sesiones persistidas sin pérdida de reproducibilidad.

---

### [FASE 8 — CONGELADA] Entrenamiento Accesible, Modular y Auditado

**Objetivo:** Reducir el coste de mejorar el modelo atacando únicamente los fallos reales detectados en el Banco de Fase 6, con contabilidad completa de recursos.

#### 3.5. Estrategia de adaptación barata y modular
1. **No preentrenar conocimiento general desde cero:** Partir de un tronco base ya estable (Fase 3 o pesos abiertos validados en Fase 4) y mantenerlo congelado como columna vertebral compartida.
2. **Especialistas modulares empaquetables:** Entrenar adaptadores o expertos aislados por dominio (Código, Matemáticas, Síntesis Documental) que puedan integrarse como páginas independientes en `Paged BitCOS` sin interferencia catastrófica entre dominios.
3. **Destilación *on-policy* + RLVR controlado:**
   - Generar trayectorias (*rollouts*) del propio estudiante, filtradas por verificadores automáticos (compilador, evaluador simbólico, verificador de citas) estrictamente separados de los tests ocultos de Fase 6.
   - Mezclar datos sintéticos con una fracción fija de datos reales verificados ($\ge 10\%$) para prevenir colapso de distribución.
4. **Validación de cuantización 1.58–2 bits:** Medir explícitamente la retención de exactitud en los 3 dominios tras cuantizar o aplicar QAT al especialista; no asumir transferencia gratuita de resultados de modelos ternarios entrenados desde cero con billones de tokens.
5. **Ablación controlada de optimizadores:** Evaluar **Muon vs. AdamW** de forma aislada según el régimen (preentrenamiento vs. *fine-tuning*), adoptando solo el que demuestre menor coste real para la misma métrica de validación.

#### 3.6. Contabilidad obligatoria de coste de entrenamiento
Toda corrida de Fase 8 debe emitir un registro JSON de coste total que incluya:
- Horas de GPU/CPU y hardware exacto.
- Coste de generación de datos sintéticos o inferencia del modelo profesor.
- Coste de generación de *rollouts* y actualizaciones de política en RLVR.
- Número y coste de **intentos fallidos** y evaluaciones intermedias.

**Puerta de salida de Fase 8:**
- Mejora verificada de **C4 sobre C3** en tests ocultos de al menos un dominio prioritario, manteniendo el perfil de latencia/RAM del motor C y con informe auditado de coste total.

---

### [FASE 9 — CONGELADA] Frontera *Tokenizer-Free* (Ablación A/B Controlada)

**Objetivo:** Evaluar si sustituir el BPE compacto por representaciones basadas en bytes aporta ventaja real en coste por token, robustez o I/O de NVMe sobre el motor TRACE ya maduro.

#### 3.7. Protocolo A/B contra la línea base BPE (mismo cómputo y hardware)
Se prohíbe mezclar entrenamiento desde cero de lenguaje + cuantización STE + segmentación discreta simultáneamente sin una línea base estable. Se comparan tres candidatos:
1. **Línea base:** BPE compacto (4K–8K, dígitos 1:1) de Fases 3–8.
2. **Candidato A (*Byteification* post-hoc):** Conversión de un tronco BPE ya estabilizado a operación sobre bytes mediante parches latentes con bajo porcentaje de cómputo adicional.
3. **Candidato B (*Chunking* jerárquico ligero, p. ej. estilo H-Net / BLT):** Segmentación dinámica aprendida evaluada exclusivamente si logra reducir llamadas al núcleo global / tráfico NVMe sin romper la estabilidad del KV-cache en `trace_infer.c` ni degradar aritmética/código.

**Puerta de salida de Fase 9:**
- Decisión basada en mediciones reales en el i3-1115G4: solo se adopta reemplazo de BPE si iguala o supera la calidad en los 3 dominios del Banco de Fase 6 y mejora la latencia real o la robustez ante ruido/código bajo el mismo presupuesto de RAM y FLOPs.

---

## 4. Cláusulas Anti-Humo (Reglas de Disciplina Epistémica)

1. No declarar «1M de tokens usable en 16 GB» basándose en cálculos parciales de latentes o benchmarks en GPUs con >1 TB/s de ancho de banda.
2. No concluir superioridad de un modelo pequeño con herramientas frente a un modelo mayor evaluado sin herramientas.
3. No afirmar que RLVR o la destilación «cuestan muy poco» sin contabilizar *rollouts*, generación del profesor e intentos fallidos.
4. No extrapolar calibración por token/dígito (`ECE`) como garantía contra alucinaciones factuales en lenguaje abierto.
5. No activar fases congeladas antes de cerrar y auditar criptográficamente (`RELEASE_MANIFEST.md`) la puerta de salida de la fase activa.
