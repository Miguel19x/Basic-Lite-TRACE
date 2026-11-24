# Política de Seguridad — Basic-Lite TRACE

## Versiones Soportadas
Actualmente se reciben reportes de seguridad para la rama activa `main` (`v1.3.0-trace` y posteriores, incluyendo el motor nativo C bajo `csrc/` con lectura `O_DIRECT`, `pread` y micro-kernels AVX2). Las versiones anteriores (`v1.0.0-trace` a `v1.2.0-trace`) se conservan únicamente como referencia histórica sin parches de seguridad.

## Reporte de Vulnerabilidades
Si descubres una vulnerabilidad potencial de seguridad (desbordamiento de búfer en kernels C, corrupción de memoria en paginación o fugas de descriptores):
1. **NO** abras una issue pública en GitHub.
2. Envía un correo con los detalles técnicos a: **miguelaranguren19x@outlook.com**.
3. Incluye una descripción detallada, prueba de concepto o traza de ejecución reproducible.
4. Recibirás acuse de recibo y evaluación técnica a la mayor brevedad posible (plazo orientativo de 72 horas).
