# Basic-Lite TRACE — Memoria de Identidad Técnica y Convergencia Arquitectónica

**Proyecto:** Basic-Lite TRACE (Evolución unificada desde el lanzamiento inicial Basic-Lite 0.1)  
**Arquitectura Canónica:** **TRACE** (*Ternary Resident Architecture for Convergent Execution*)  
**Linaje Histórico:** Familias TRDM (TRDM-v1 a TRDM-v6) integradas en la arquitectura TRACE  
**Fundador y Desarrollador Principal:** Miguel19x  
**Contacto oficial:** `miguelaranguren19x@outlook.com`  
**Repositorio oficial:** [https://github.com/Miguel19x/Basic-Lite-TRACE](https://github.com/Miguel19x/Basic-Lite-TRACE)  

---

## 1. La Idea Central y Autoría de la Integración

La concepción, el diseño y la integración de todo este modelo y sistema en una arquitectura única y funcional **proviene íntegramente de Miguel19x**.

En la computación contemporánea de modelos de lenguaje (>10B - 30B parámetros), existe una barrera de hardware crítica: se requieren decenas de gigabytes de memoria VRAM de alta velocidad en aceleradores GPU/TPU dedicados o clústeres masivos. En un procesador doméstico estándar (**Intel Core i3-1115G4**, 2 núcleos físicos / 4 hilos, 16 GB RAM, SSD NVMe, Linux x86_64 sin GPU), cualquier modelo denso convencional colapsa por agotamiento de memoria (OOM) o sufre degradación severa por intercambio en disco (*swap thrashing*).

Frente a esta limitación, **Miguel19x formuló la tesis central de Basic-Lite:**
> *¿Es posible desacoplar la capacidad total de almacenamiento de un modelo a gran escala (hasta 30B en disco) de la memoria RAM activa residente mediante residencia selectiva, paginación binaria estricta a nivel de sistema operativo y un ruteo MoE penalizado por la latencia física de acceso a almacenamiento secundario NVMe?*

Para materializar esta visión nació la arquitectura **TRACE** y el sistema **Basic-Lite TRACE**.

---

## 2. Linaje Evolutivo Completo: De Basic-Lite 0.1 a Basic-Lite TRACE

El proyecto representa una línea de investigación experimental continua sin fracturas ni archivados: lo que nació y se publicó como **Basic-Lite 0.1** evoluciona de manera unificada y directa hacia **Basic-Lite TRACE**, combinando la totalidad de los commits, hitos y desarrollos en este repositorio único:

### 2.1. Fase Inicial: Génesis en Basic-Lite 0.1 (TRDM-v1, v2, v3 y v4)
* **Punto de Partida:** Nacimiento del proyecto con el commit inicial `aa7ce9e` (14 de noviembre de 2026).
* **Exploración de Cuantización:** Adopción de la cuantización ternaria pura W1.58A8 con Straight-Through Estimator (STE).
* **Kernels en C Nativos:** Creación del primer kernel SIMD de suma/resta directa sin unidades de multiplicación flotante (`ternary_kernel.c`).
* **Empaquetado binario:** Primer formato `bitcos.bin` y hallazgos diagnósticos sobre el sobreajuste de longitud fija.

### 2.2. Fase de Consolidación y Convergencia: Hacia la Arquitectura TRACE
* **Evaluación Confirmatoria de Fase 1:** Definición de criterios reproducibles, higiene E0 y evaluación multi-semilla (etiqueta histórica `prereg-v1`).
* **Paged BitCOS v1:** Paginación de tensores W1.58 alineada estrictamente a 4096 bytes del kernel Linux ([trace/serialization/paged_bitcos.py](../trace/serialization/paged_bitcos.py)) y kernel C fusionado ([paged_gemm_kernel.c](../paged_gemm_kernel.c)).
* **Enrutamiento MoE Consciente de NVMe:** Ruteo penalizado por latencia de E/S con *buddy routing* ([trace/modules/paged_moe.py](../trace/modules/paged_moe.py)).
* **Campo Causal Desacoplado en 2 Etapas con `detach()`:** Reemplazo de la difusión discreta pura por el campo causal en dos etapas ([trace/models/causal.py](../trace/models/causal.py)), resolviendo la multiplicación al $81.88\%$ y rompiendo la nulidad en Full-EM ($5.12\% \pm 4.88\%$).
* **TRACE:** Bautismo formal de la arquitectura convergente unificada: **T**ernary **R**esident **A**rchitecture for **C**onvergent **E**xecution.

---

## 3. Estado y Funcionamiento de los Componentes en la Arquitectura TRACE

En la arquitectura convergente de **Basic-Lite TRACE**, el modelo responde plenamente a sus pilares fundacionales de cuantización y recurrencia, con una evolución crítica en el mecanismo de generación:

* **Ternary (Ternario W1.58A8 — Preservado al 100%):**  
  Todas las proyecciones nucleares operan con pesos $\{-1, 0, 1\}$ mediante [trace/layers/bitlinear.py](../trace/layers/bitlinear.py). En runtime, las matrices se evalúan directamente sobre las páginas binarias mapeadas en memoria sin descomprimir a FP32 en RAM.

* **Recursive (Recursividad Dinámica — Preservado y Reforzado):**  
  El núcleo residente compartido ([trace/modules/resident_core.py](../trace/modules/resident_core.py)) no apila capas densas tradicionales, sino que reitera $R$ pasos de recursión sobre un estado latente dual ($z_r$ para razonamiento profundo, $y_r$ para respuesta). Incorpora codificación sinusoidal continua de profundidad $e_r$ y compuerta adaptativa de parada temprana (*adaptive halting gate*).

* **Convergent / Causal (Campo Causal Desacoplado en 2 Etapas):**  
  * *Antecedente:* Históricamente se evaluó desruidado por difusión discreta, pero el desenmascaramiento paralelo colapsaba en aritmética composicional ($0.04\% \pm 0.09\%$ Full-EM).
  * *Mecanismo TRACE:* El mecanismo de generación en producción es el **Campo Causal Desacoplado en 2 Etapas con `detach()`** ([trace/models/causal.py](../trace/models/causal.py)). Este desacople resolvió los subtotales multiplicativos al **$81.88\%$** y rompió la barrera de nulidad en Exact Match ($5.12\% \pm 4.88\%$, con pico de $13.33\%$).

---

## 4. Todo lo que se Utiliza para que el Sistema Funcione (Los 5 Ejes de TRACE)

TRACE opera dentro del ecosistema de residencia selectiva a través de **cinco ejes canónicos interdependientes**:

1. **Paged BitCOS v1 ([trace/serialization/paged_bitcos.py](../trace/serialization/paged_bitcos.py) y [paged_gemm_kernel.c](../paged_gemm_kernel.c)):**  
   Paginación de tensores en bloques alineados a 4096 bytes del kernel Linux consumidos vía `mmap`.
2. **MoE Consciente de Memoria Física ([trace/modules/paged_moe.py](../trace/modules/paged_moe.py)):**  
   Ruteo dinámico con penalización por I/O físico: $S(e) = U(e) - \lambda C_{\text{lectura}} - \mu C_{\text{cómputo}}$, respaldado por expertos sustitutos (*buddy routing*).
3. **MLA + Gestor de Contexto en 3 Niveles ([trace/modules/mla_attention.py](../trace/modules/mla_attention.py) y [trace/core/context_manager.py](../trace/core/context_manager.py)):**  
   Compresión latente de claves y valores (KV 4×–8×) en RAM activa (L1), resúmenes latentes (L2) y almacenamiento estructurado en SQLite sobre disco (L3).
4. **Hierarchical Byte Patching ($P=4$, [trace/layers/byte_patcher.py](../trace/layers/byte_patcher.py)):**  
   Reducción cuadrática de $16\times$ del coste de atención sobre secuencias crudas de bytes sin tokenizador rígido.
5. **Abacus Positional Embeddings + Campo Causal en 2 Etapas ([trace/layers/abacus.py](../trace/layers/abacus.py) y [trace/models/causal.py](../trace/models/causal.py)):**  
   Conciencia posicional decimal de derecha a izquierda y corte de gradiente para eliminar el sesgo de exposición (*exposure bias*).

---

## 5. Resultados Empíricos Medidos vs. Proyección

* **Hecho medido y reproducible en Intel Core i3-1115G4 (5 semillas oficiales):**  
  Validez sintáctica del $100.00\%$, resolución de subtotales al $81.88\% \pm 10.14\%$ (máximo $95.00\%$), y Exact Match de $5.12\% \pm 4.88\%$ (máximo $13.33\%$).
* **Línea abierta de investigación:**  
  La propagación multivariable de acarreos en la suma final de la Etapa 2 constituye el desafío activo a resolver.
* **Proyección de diseño a escala (30B):**  
  Extender esta arquitectura completa a modelos de 30B parámetros sobre 16 GB de RAM es la hipótesis de ingeniería que guía las siguientes fases del desarrollo.

---

## 6. Contacto y Colaboraciones

* **Investigadores y Académicos:** El código está disponible para estudio, auditoría y contraste empírico bajo los términos de [LICENSE.txt](../LICENSE.txt).
* **Empresas e Integradores:** Para evaluaciones de viabilidad en hardware local o adaptaciones con soporte directo del autor:  
  **Miguel19x** — `miguelaranguren19x@outlook.com`
