# Directrices de Colaboración — Basic-Lite TRACE

Gracias por tu interés en **Basic-Lite TRACE**.

Para preservar la trazabilidad científica, la coherencia arquitectónica del motor C y la titularidad legal única del proyecto, **Basic-Lite TRACE** sigue un modelo de desarrollo similar al de **SQLite**: código visible y auditable, con el núcleo gestionado de forma centralizada y **tres canales claros de participación**.

---

## 1. Los Tres Canales de Colaboración

### Canal 1 — Benchmarks Comunitarios y Reproducción en Hardware Diverso *(Abierto vía Issues)*
Si dispones de hardware distinto a la máquina base de referencia (`Intel Core i3-1115G4`) —por ejemplo **AMD Ryzen, Intel AVX-512, ARM64 / Apple Silicon o servidores Linux**—, te invitamos a ejecutar `make all`, `make test` o `./trace_lm` y abrir una **Issue** con tus mediciones (TTFT, `tok/s`, RAM pico `Max RSS` y versión de GCC/kernel). Esta validación empírica externa fortalece directamente la evidencia del proyecto.

### Canal 2 — Reportes de Errores y Diagnósticos Técnicos *(Abierto vía Issues)*
Puedes abrir una **Issue** para reportar fallos de compilación, discrepancias numéricas o análisis de cuellos de botella (por ejemplo, optimización del *prefill* en frío en `csrc/lm/trace_lm_infer.c` o soporte multi-arquitectura). Para mantener limpia la cadena de propiedad intelectual (*clean-room*), las correcciones en el núcleo se integran internamente a partir del diagnóstico documentado.

### Canal 3 — Incorporación al Equipo Núcleo *(Acuerdo Bilateral Formal)*
El proyecto busca colaboradores técnicos de alto nivel interesados en construir a largo plazo en:
* **Sistemas y Bajo Nivel (C / SIMD / Linux):** Kernels ternarios (AVX2/AVX-512/NEON), `O_DIRECT` / `io_uring` y perfilado de latencia.
* **Investigación y Escalado ML (PyTorch):** Entrenamiento QAT ternario, MLA, MoE paginado y evaluación rigurosa.

> [!TIP]
> **La mejor puerta de entrada al equipo núcleo** es abrir primero una Issue técnica de calidad o publicar una reproducción independiente; a partir de ahí coordinamos directamente un acuerdo formal de coautoría científica o colaboración escribiendo a `miguelaranguren19x@outlook.com`.

---

## 2. Política de Pull Requests

* **Sin Pull Requests abiertos no acordados:** Actualmente no se fusionan *pull requests* externos sin coordinación previa ni acuerdo formal de coautoría o cesión de derechos, con el fin de evitar ambigüedades de titularidad o contaminación de licencias.
* **Bifurcaciones (*forks*) independientes:** Cualquier persona u organización puede bifurcar el repositorio y explorar modificaciones técnicas de manera independiente para los fines autorizados, cumpliendo las condiciones de atribución, licencia y entrega del código fuente establecidas en [LICENSE.txt](LICENSE.txt).

---

## 3. Reglas de Privacidad y Seguridad en Issues

Al abrir una Issue técnica o compartir resultados de benchmark:
* **Privacidad de datos:** No envíes datos personales, confidenciales, credenciales ni conjuntos de datos propietarios en issues o *logs*. Utiliza siempre ejemplos mínimos y sintéticos.
* **Vulnerabilidades de seguridad:** Si descubres un fallo de memoria en C, corrupción de paginación o fuga de descriptores, **NO abras una issue pública**; sigue el canal privado de [SECURITY.md](SECURITY.md).
* **Alcance de la atención:** Esta publicación no incluye soporte técnico garantizado ni validación de aplicaciones finales salvo acuerdo bilateral separado.
