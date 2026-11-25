# Basic-Lite TRACE — Avisos de Componentes de Terceros (Third Party Notices)

Este proyecto hace uso de librerías y componentes de software de código abierto bajo sus respectivas licencias. Este documento compila las atribuciones y términos exigidos por dichas licencias de conformidad con los estándares de buenas prácticas de la OMPI.

---

## 1. Inventario de Dependencias Externas

| Componente | Proyecto / Procedencia | Tipo de Licencia | Uso en Basic-Lite | Archivo de Licencia |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch** | PyTorch Foundation / Meta AI / The Linux Foundation | BSD-3-Clause | Tensores base, autograd, capas neuronales y wrappers | `licenses/LICENSE_PYTORCH.txt` |
| **NumPy** | NumPy Developers | BSD-3-Clause | Operaciones numéricas, generación de arrays y evaluación | `licenses/LICENSE_NUMPY.txt` |
| **SQLite3** | SQLite Consortium (D. Richard Hipp) | Dominio Público / SQLite Blessing | Persistencia en disco del Nivel 3 (`trace/core/context_manager.py`) y enciclopedia factual offline FTS5 (`trace/tir/l3_store.py`) | `licenses/LICENSE_SQLITE.txt` |
| **Linux POSIX / mmap** | Linux Kernel Organization / POSIX standards | GPLv2 (Kernel) / Interfaz POSIX de usuario (Libc) | Mapeo directo a páginas de memoria del OS (`O_DIRECT` + `pread`) en Paged BitCOS | Headers del sistema operativo |
| **OpenMP** | OpenMP Architecture Review Board | Runtime de compilador GCC/Clang (GCC Runtime Library Exception) | Paralelismo multinúcleo en kernels C optimizados (AVX-512 VNNI / AVX2 / Escalar) | Headers del sistema |
| **pytest** *(solo desarrollo/CI)* | pytest-dev | MIT License | Ejecución de suites de pruebas automatizadas en CI | Instalado vía pip en CI |
| **Ruff** *(solo desarrollo/CI)* | Astral Software | MIT License | Verificación estática de estilo y sintaxis Python en CI | Instalado vía pip en CI |

---

## 2. Textos Completos de Licencias de Terceros

Los textos completos de las licencias de código abierto se conservan íntegramente en la carpeta `licenses/`:

1. **PyTorch License:** [licenses/LICENSE_PYTORCH.txt](licenses/LICENSE_PYTORCH.txt)
2. **NumPy License:** [licenses/LICENSE_NUMPY.txt](licenses/LICENSE_NUMPY.txt)
3. **SQLite License & Blessing:** [licenses/LICENSE_SQLITE.txt](licenses/LICENSE_SQLITE.txt)

---

## 3. Declaración de Independencia, Modelos Externos y No Afectación

* Los componentes de terceros se rigen estrictamente por sus términos de distribución originales.
* El aviso de reserva de derechos y protección legal de **Basic-Lite TRACE** contenido en `LICENSE.txt` y `NOTICE.md` aplica **únicamente** a los módulos, algoritmos, kernels, pesos entrenados y artefactos de autoría original desarrollados por Miguel19x, y no pretende en modo alguno atribuirse titularidad sobre librerías, sistemas operativos ni modelos o pesos abiertos de terceros (por ejemplo, pesos externos compatibles como `microsoft/BitNet-b1.58-2B-4T` bajo licencia MIT u otros modelos abiertos que el usuario decida convertir o ejecutar mediante el formato Paged BitCOS).
* Cuando un usuario importe, convierta o ejecute pesos de terceros en el motor Basic-Lite TRACE, dichos pesos continúan sujetos íntegramente a la licencia de su proveedor original.
