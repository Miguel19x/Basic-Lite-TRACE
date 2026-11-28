# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Banco de Evaluación de 4 Dominios + Matriz Simétrica C1-C3 + Calibración (Fase 4B).

Dominios cubiertos:
  1. code_terminal: lectura/ejecución/depuración de código en sandbox.
  2. math_logic: razonamiento lógico-matemático con resultado exacto.
  3. long_docs: localización puntual (subtask='needle') separada de síntesis multi-pasaje (subtask='synthesis').
  4. factual: historia/geografía verificable con enciclopedia offline SQLite L3 (FTS5).

Condiciones simétricas (mismo presupuesto de tokens por tarea):
  - C1: Modelo solo.
  - C2: Modelo + contexto recuperado de SQLite L3 (FTS5).
  - C3: Modelo + SQLite L3 + capa TIR con llamadas JSON estructuradas bajo Sandbox.

Calibración estrictamente desacoplada:
  - Token-level ECE (Expected Calibration Error sobre probabilidades por token).
  - Response-level Brier score (error cuadrático medio sobre exactitud de respuesta completa).
"""

from __future__ import annotations

import json
import math
import re
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from trace.tir.l3_store import L3Encyclopedia
from trace.tir.sandbox import Sandbox, SandboxPolicy
from trace.tir.tools import TIRToolRegistry

Condition = Literal["C1", "C2", "C3"]
Domain = Literal["code_terminal", "math_logic", "long_docs", "factual"]
Language = Literal["es", "en"]


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    domain: Domain
    subtask: str
    prompt: str
    expected_answer: str
    sandbox_files: dict[str, str] = field(default_factory=dict)
    l3_query: str = ""
    hint_tool: str = ""
    hint_args: dict[str, Any] = field(default_factory=dict)
    language: Language = "es"
    is_unanswerable: bool = False


@dataclass
class GenerationResult:
    text: str
    token_confidences: list[float] = field(default_factory=list)
    token_correctness: list[int] = field(default_factory=list)
    response_confidence: float | None = None
    tokens_generated: int = 0


class ModelBackend(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        json_constrained: bool = False,
    ) -> GenerationResult: ...


def compute_token_ece(
    confidences: list[float],
    correctness: list[int],
    n_bins: int = 10,
) -> float:
    """Calcula Expected Calibration Error (ECE) a nivel de token."""
    if len(confidences) != len(correctness):
        raise ValueError("confidences y correctness deben tener la misma longitud")
    if not confidences:
        return 0.0
    if n_bins <= 0:
        raise ValueError("n_bins debe ser > 0")

    n = len(confidences)
    ece = 0.0
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        in_bin = [
            i
            for i, c in enumerate(confidences)
            if (lo <= c <= hi if b == n_bins - 1 else lo <= c < hi)
        ]
        if not in_bin:
            continue
        avg_conf = sum(min(1.0, max(0.0, float(confidences[i]))) for i in in_bin) / len(in_bin)
        avg_acc = sum(1.0 if int(correctness[i]) else 0.0 for i in in_bin) / len(in_bin)
        ece += (len(in_bin) / n) * abs(avg_acc - avg_conf)
    return float(ece)


def compute_response_brier(probs: list[float], outcomes: list[int]) -> float:
    """Calcula puntuación Brier a nivel de respuesta completa: (1/N) * sum((p_i - y_i)^2)."""
    if len(probs) != len(outcomes):
        raise ValueError("probs y outcomes deben tener la misma longitud")
    if not probs:
        return 0.0
    total = 0.0
    for p, y in zip(probs, outcomes):
        pc = min(1.0, max(0.0, float(p)))
        yc = 1.0 if int(y) else 0.0
        total += (pc - yc) ** 2
    return float(total / len(probs))


def _derive_response_confidence(res: GenerationResult) -> float:
    if res.response_confidence is not None:
        return min(1.0, max(0.0, float(res.response_confidence)))
    if not res.token_confidences:
        return 0.5
    # Media geométrica estable de probabilidades por token
    log_sum = sum(math.log(max(1e-6, min(1.0, float(p)))) for p in res.token_confidences)
    return float(math.exp(log_sum / len(res.token_confidences)))


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def count_think_tokens(text: str) -> int:
    """
    Cuenta de forma determinista los tokens delimitados por espacios dentro de todos los bloques
    `<think>...</think>` (insensible a mayúsculas) e incluye cualquier bloque `<think>` abierto
    sin cierre al final (p. ej. por truncamiento de presupuesto). Devuelve 0 si no hay bloques.
    """
    closed_re = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
    total = sum(len(block.split()) for block in closed_re.findall(text))
    remainder = closed_re.sub("", text)
    unclosed = re.search(r"<think>(.*)$", remainder, re.DOTALL | re.IGNORECASE)
    if unclosed is not None:
        total += len(unclosed.group(1).split())
    return total


def grade_answer(
    response_text: str,
    expected_answer: str,
    *,
    is_unanswerable: bool = False,
) -> bool:
    """
    Evalúa coincidencia exacta del valor semántico esperado sin inflar por plantilla.
    Si existe etiqueta `<answer>...</answer>` (Fase 5, insensible a mayúsculas), evalúa
    únicamente el último bloque `<answer>`. Sin etiqueta `<answer>`, elimina todos los bloques
    `<think>...</think>` y cualquier `<think>` truncado sin cierre antes de evaluar.
    Exige que el valor esperado aparezca como token/número delimitado (no subcadena accidental,
    ni número negativo/decimal distinto como `-70` o `70.5` para `70`).
    Cuando `is_unanswerable=True`, rechaza negaciones del marcador de abstención o respuestas
    numéricas contradictorias adjuntas.
    """
    exp = _normalize_text(expected_answer)
    if not exp:
        return False
    stripped = re.sub(r"<think>.*?</think>", " ", response_text, flags=re.DOTALL | re.IGNORECASE)
    stripped = re.sub(r"<think>.*$", " ", stripped, flags=re.DOTALL | re.IGNORECASE)
    ans_blocks = re.findall(r"<answer>(.*?)</answer>", stripped, re.DOTALL | re.IGNORECASE)
    candidate = ans_blocks[-1] if ans_blocks else stripped
    norm_resp = _normalize_text(candidate)
    pattern = rf"(?<![0-9a-zA-Z_\-])(?<![0-9]\.){re.escape(exp)}(?![0-9a-zA-Z_\-]|\.[0-9])"
    matched = norm_resp == exp or re.search(pattern, norm_resp) is not None
    if not matched:
        return False
    if is_unanswerable:
        neg_pattern = (
            rf"\b(?:not|no|no\s+es|no\s+hay|is\s+not|isn't|don't|do\s+not|never|nunca)"
            rf"\s+(?:[a-záéíóúñ]+\s+)?{re.escape(exp)}\b"
        )
        if re.search(neg_pattern, norm_resp) is not None:
            return False
        contradictory_answer_re = (
            r"(?:"
            r"\b(?:the\s+answer|answer|result|value|la\s+respuesta|respuesta|resultado|valor)"
            r"\s*(?:is|es|=|:)\s*-?\d+(?:\.\d+)?"
            r"|\b(?:is|es)\s+-?\d+(?:\.\d+)?\b"
            r"|=\s*-?\d+(?:\.\d+)?"
            rf"|{re.escape(exp)}\s*[,;:]\s*-?\d+(?:\.\d+)?\b"
            r")"
        )
        if re.search(contradictory_answer_re, norm_resp) is not None:
            return False
    return True


def build_default_l3_corpus() -> list[tuple[str, str, str]]:
    """Corpus factual offline determinista para poblar SQLite L3 (FTS5)."""
    return [
        (
            "Tratado de Tordesillas",
            "history",
            "El Tratado de Tordesillas fue firmado el 7 de junio de 1494 entre Castilla y Portugal "
            "para establecer una línea de demarcación en el océano Atlántico.",
        ),
        (
            "Lago Baikal",
            "geography",
            "El lago Baikal en Siberia, Rusia, es el lago más profundo de la Tierra con una "
            "profundidad máxima verificada de 1642 metros.",
        ),
        (
            "Constante de Avogadro",
            "science",
            "Desde la redefinición del SI de 2019, la constante de Avogadro tiene el valor exacto "
            "de 6.02214076e23 mol^-1.",
        ),
        (
            "Canal de Panamá",
            "geography",
            "El Canal de Panamá fue inaugurado oficialmente el 15 de agosto de 1914 y conecta "
            "el océano Atlántico con el océano Pacífico.",
        ),
    ]


def build_default_tasks() -> list[BenchmarkTask]:
    """Devuelve la suite oculta determinista de 4 dominios (8 tareas, 2 por dominio)."""
    distractor_a = (
        "Registro operativo estándar: el módulo de telemetría muestrea cada 500 ms en modo nominal "
        "y almacena resúmenes diarios en formato binario compacto sin compresión externa. "
    ) * 4
    distractor_b = (
        "Nota de mantenimiento: todos los conectores de fibra óptica deben limpiarse antes de cada "
        "inspección trimestral y etiquetarse con el código de bahía correspondiente. "
    ) * 4

    needle_doc = (
        f"{distractor_a}\n"
        "CLAVE_AUDITORIA_INTERNA: DELTA-7492 (válida solo para el lote de verificación).\n"
        f"{distractor_b}\n"
        "Pregunta: ¿Cuál es exactamente la CLAVE_AUDITORIA_INTERNA indicada en el documento? "
        "Responde únicamente con el código exacto."
    )

    synthesis_doc = (
        "Sección 1 (Regla Base): Para la estación térmica serie R4, la temperatura máxima permitida "
        "en operación continua es de 85 C a nivel del mar.\n"
        f"{distractor_a}\n"
        "Sección 2 (Modificación por Altitud): Cuando una unidad de la serie R4 opera por encima "
        "de 2000 metros de altitud, la temperatura máxima permitida se reduce obligatoriamente en 15 C "
        "respecto a la regla base.\n"
        f"{distractor_b}\n"
        "Pregunta: ¿Cuál es la temperatura máxima permitida (en grados C, solo el número) para una "
        "estación serie R4 desplegada a 2400 metros de altitud?"
    )

    return [
        # 1. Code / Terminal / Files
        BenchmarkTask(
            task_id="code_py_exec_1",
            domain="code_terminal",
            subtask="python_sandbox_exec",
            prompt=(
                "En el workspace local existe el archivo `sum_squares.py`. "
                "Ejecútalo en el sandbox (o léelo y calcula su salida) y devuelve únicamente "
                "el entero exacto que imprime."
            ),
            expected_answer="2870",
            sandbox_files={
                "sum_squares.py": "print(sum(i * i for i in range(1, 21)))\n",
            },
            hint_tool="run_cmd",
            hint_args={"argv": ["python3", "sum_squares.py"]},
        ),
        BenchmarkTask(
            task_id="code_file_inspect_2",
            domain="code_terminal",
            subtask="file_read_inspect",
            prompt=(
                "Lee el archivo `config_target.txt` en el workspace local y responde únicamente "
                "con el valor numérico asignado a `TARGET_PORT`."
            ),
            expected_answer="9443",
            sandbox_files={
                "config_target.txt": "HOST=127.0.0.1\nTARGET_PORT=9443\nMODE=offline\n",
            },
            hint_tool="read_file",
            hint_args={"path": "config_target.txt"},
        ),
        # 2. Logical-Mathematical
        BenchmarkTask(
            task_id="math_exact_1",
            domain="math_logic",
            subtask="exact_arithmetic",
            prompt=(
                "Calcula el valor entero exacto de `(347 * 829) - (193 * 412) + (7 ** 4)`. "
                "Responde únicamente con el número entero."
            ),
            expected_answer="210548",
            hint_tool="calc",
            hint_args={"expr": "(347 * 829) - (193 * 412) + (7 ** 4)"},
        ),
        BenchmarkTask(
            task_id="math_logic_2",
            domain="math_logic",
            subtask="gcd_factorial",
            prompt=(
                "Calcula el valor entero exacto de `gcd(4620, 3570) * 19 + factorial(6)`. "
                "Responde únicamente con el número entero."
            ),
            expected_answer="4710",
            hint_tool="calc",
            hint_args={"expr": "gcd(4620, 3570) * 19 + factorial(6)"},
        ),
        # 3. Long Documents (needle separated from multi-passage synthesis)
        BenchmarkTask(
            task_id="docs_needle_1",
            domain="long_docs",
            subtask="needle",
            prompt=needle_doc,
            expected_answer="DELTA-7492",
        ),
        BenchmarkTask(
            task_id="docs_synthesis_1",
            domain="long_docs",
            subtask="synthesis",
            prompt=synthesis_doc,
            expected_answer="70",
            hint_tool="calc",
            hint_args={"expr": "85 - 15"},
        ),
        # 4. Factual Knowledge (paired with offline SQLite L3)
        BenchmarkTask(
            task_id="fact_hist_1",
            domain="factual",
            subtask="history",
            prompt=(
                "¿En qué año se firmó el Tratado de Tordesillas entre Castilla y Portugal? "
                "Responde únicamente con el año de 4 dígitos."
            ),
            expected_answer="1494",
            l3_query="Tratado de Tordesillas",
            hint_tool="l3_search",
            hint_args={"query": "Tratado de Tordesillas", "top_k": 1},
        ),
        BenchmarkTask(
            task_id="fact_geo_2",
            domain="factual",
            subtask="geography",
            prompt=(
                "¿Cuál es la profundidad máxima verificada en metros del lago Baikal? "
                "Responde únicamente con el número entero en metros."
            ),
            expected_answer="1642",
            l3_query="profundidad lago Baikal",
            hint_tool="l3_search",
            hint_args={"query": "lago Baikal profundidad", "top_k": 1},
        ),
    ]


def build_extended_l3_corpus() -> list[tuple[str, str, str]]:
    """
    Corpus factual offline determinista para la suite extendida (12 hechos disjuntos de
    `build_default_l3_corpus` y del generador de entrenamiento de Fase 5).
    """
    return [
        (
            "Magna Carta",
            "history",
            "Magna Carta was sealed by King John of England at Runnymede in 1215.",
        ),
        (
            "Mount Fuji",
            "geography",
            "Mount Fuji on Honshu island in Japan has a summit elevation of 3776 meters.",
        ),
        (
            "Suez Canal",
            "history",
            "The Suez Canal officially opened to international navigation in 1869.",
        ),
        (
            "Iron Element",
            "science",
            "Iron (chemical symbol Fe) has atomic number 26 on the periodic table.",
        ),
        (
            "CERN Founding",
            "science",
            "The European Organization for Nuclear Research (CERN) was formally established in 1954.",
        ),
        (
            "Penicillin Discovery",
            "science",
            "Alexander Fleming discovered penicillin in London in 1928.",
        ),
        (
            "Paz de Westfalia",
            "history",
            "La Paz de Westfalia que puso fin a la Guerra de los Treinta Años se firmó en 1648.",
        ),
        (
            "Monte Aconcagua",
            "geography",
            "El Monte Aconcagua en los Andes argentinos alcanza una altitud oficial de 6961 metros.",
        ),
        (
            "Monte K2",
            "geography",
            "El pico K2 en la cordillera del Karakórum tiene una altitud verificada de 8611 metros.",
        ),
        (
            "Elemento Cobre",
            "science",
            "El cobre (símbolo Cu) posee el número atómico 29 en la tabla periódica.",
        ),
        (
            "Cable Telegráfico Transatlántico",
            "history",
            "El primer cable telegráfico transatlántico operativo transmitió mensajes en 1858.",
        ),
        (
            "Constitución de Cádiz",
            "history",
            "La Constitución Española de Cádiz fue promulgada por las Cortes en 1812.",
        ),
    ]


def build_extended_suite() -> list[BenchmarkTask]:
    """
    Devuelve la suite oculta extendida y determinista de 4 dominios (N=64):
      - 16 tareas por dominio (`code_terminal`, `math_logic`, `long_docs`, `factual`).
      - 50% EN / 50% ES (8 EN + 8 ES por dominio).
      - Incluye casos de abstención honesta (`is_unanswerable=True`) para calibración Brier.
      - 0% de fuga respecto a `build_default_tasks()` y al generador de Fase 5.
    """
    filler_en_1 = (
        "Routine telemetry log: sensor calibration drift remained within 0.02 percent across all "
        "passive thermal channels during the scheduled night shift without manual intervention. "
    ) * 3
    filler_en_2 = (
        "Compliance appendix: rack air filters are inspected monthly and archived under standard "
        "facility procedures with no impact on active compute quotas. "
    ) * 3
    filler_es_1 = (
        "Bitácora de respaldo: los ventiladores auxiliares operaron a velocidad nominal durante el "
        "turno nocturno sin alertas térmicas ni interrupciones en el bus secundario. "
    ) * 3
    filler_es_2 = (
        "Anexo operativo: el inventario de cables blindados se revisa cada trimestre conforme al "
        "manual interno de almacenamiento sin modificar parámetros de red. "
    ) * 3

    tasks: list[BenchmarkTask] = [
        # =====================================================================
        # 1. code_terminal (16: 8 EN + 8 ES)
        # =====================================================================
        BenchmarkTask(
            task_id="ext_code_en_01",
            domain="code_terminal",
            subtask="python_sandbox_exec",
            language="en",
            prompt=(
                "Run `cube_sum.py` in the local sandbox workspace and reply with only the exact integer printed."
            ),
            expected_answer="3025",
            sandbox_files={"cube_sum.py": "print(sum(n ** 3 for n in range(1, 11)))\n"},
            hint_tool="run_cmd",
            hint_args={"argv": ["python3", "cube_sum.py"]},
        ),
        BenchmarkTask(
            task_id="ext_code_en_02",
            domain="code_terminal",
            subtask="python_sandbox_exec",
            language="en",
            prompt=(
                "Execute `even_filter.py` in the workspace and return only the integer output."
            ),
            expected_answer="90",
            sandbox_files={
                "even_filter.py": "vals = [14, 7, 22, 9, 36, 5, 18]\nprint(sum(v for v in vals if v % 2 == 0))\n"
            },
            hint_tool="run_cmd",
            hint_args={"argv": ["python3", "even_filter.py"]},
        ),
        BenchmarkTask(
            task_id="ext_code_en_03",
            domain="code_terminal",
            subtask="python_sandbox_exec",
            language="en",
            prompt=(
                "Execute `word_count.py` in the local workspace and output only the printed integer."
            ),
            expected_answer="3",
            sandbox_files={
                "word_count.py": "text = 'alpha beta gamma alpha delta alpha'\nprint(text.split().count('alpha'))\n"
            },
            hint_tool="run_cmd",
            hint_args={"argv": ["python3", "word_count.py"]},
        ),
        BenchmarkTask(
            task_id="ext_code_en_04",
            domain="code_terminal",
            subtask="python_sandbox_exec",
            language="en",
            prompt=(
                "Run `fib_step.py` in the sandbox and respond solely with the exact integer printed."
            ),
            expected_answer="144",
            sandbox_files={
                "fib_step.py": "a, b = 0, 1\nfor _ in range(12):\n    a, b = b, a + b\nprint(a)\n"
            },
            hint_tool="run_cmd",
            hint_args={"argv": ["python3", "fib_step.py"]},
        ),
        BenchmarkTask(
            task_id="ext_code_en_05",
            domain="code_terminal",
            subtask="file_read_inspect",
            language="en",
            prompt=(
                "Inspect `gateway.ini` in the workspace and reply with only the numeric value of `TIMEOUT_MS`."
            ),
            expected_answer="4250",
            sandbox_files={
                "gateway.ini": "[network]\nRETRY_LIMIT=7\nCACHE_MB=256\nTIMEOUT_MS=4250\n"
            },
            hint_tool="read_file",
            hint_args={"path": "gateway.ini"},
        ),
        BenchmarkTask(
            task_id="ext_code_en_06",
            domain="code_terminal",
            subtask="file_read_inspect",
            language="en",
            prompt=(
                "Read `release_meta.env` in the workspace and return only the integer assigned to `SCHEMA_REV`."
            ),
            expected_answer="19",
            sandbox_files={
                "release_meta.env": "BUILD_CHANNEL=canary\nSCHEMA_REV=19\nWORKERS=4\n"
            },
            hint_tool="read_file",
            hint_args={"path": "release_meta.env"},
        ),
        BenchmarkTask(
            task_id="ext_code_en_07",
            domain="code_terminal",
            subtask="file_read_inspect",
            language="en",
            prompt=(
                "Read `metrics_snap.json` in the workspace and return only the integer value of `queue_depth`."
            ),
            expected_answer="83",
            sandbox_files={
                "metrics_snap.json": '{"node": "west-2", "active_shards": 14, "queue_depth": 83}\n'
            },
            hint_tool="read_file",
            hint_args={"path": "metrics_snap.json"},
        ),
        BenchmarkTask(
            task_id="ext_code_en_08",
            domain="code_terminal",
            subtask="python_sandbox_exec",
            language="en",
            prompt=(
                "Run `range_span.py` in the local sandbox and reply with only the exact integer output."
            ),
            expected_answer="265",
            sandbox_files={
                "range_span.py": "rows = [120, 45, 310, 85]\nprint(max(rows) - min(rows))\n"
            },
            hint_tool="run_cmd",
            hint_args={"argv": ["python3", "range_span.py"]},
        ),
        BenchmarkTask(
            task_id="ext_code_es_01",
            domain="code_terminal",
            subtask="python_sandbox_exec",
            language="es",
            prompt=(
                "Ejecuta `suma_impares.py` en el sandbox local y responde únicamente con el entero impreso."
            ),
            expected_answer="400",
            sandbox_files={"suma_impares.py": "print(sum(x for x in range(1, 40, 2)))\n"},
            hint_tool="run_cmd",
            hint_args={"argv": ["python3", "suma_impares.py"]},
        ),
        BenchmarkTask(
            task_id="ext_code_es_02",
            domain="code_terminal",
            subtask="python_sandbox_exec",
            language="es",
            prompt=(
                "Ejecuta `conteo_vocales.py` en el workspace y devuelve únicamente el entero exacto."
            ),
            expected_answer="14",
            sandbox_files={
                "conteo_vocales.py": (
                    "s = 'arquitectura modular determinista'\n"
                    "print(sum(1 for c in s if c in 'aeiou'))\n"
                )
            },
            hint_tool="run_cmd",
            hint_args={"argv": ["python3", "conteo_vocales.py"]},
        ),
        BenchmarkTask(
            task_id="ext_code_es_03",
            domain="code_terminal",
            subtask="python_sandbox_exec",
            language="es",
            prompt=(
                "Corre `producto_pares.py` en el sandbox y entrega solo el número entero resultante."
            ),
            expected_answer="384",
            sandbox_files={
                "producto_pares.py": "res = 1\nfor n in [2, 4, 6, 8]:\n    res *= n\nprint(res)\n"
            },
            hint_tool="run_cmd",
            hint_args={"argv": ["python3", "producto_pares.py"]},
        ),
        BenchmarkTask(
            task_id="ext_code_es_04",
            domain="code_terminal",
            subtask="python_sandbox_exec",
            language="es",
            prompt=(
                "Ejecuta `extremos_lista.py` en el workspace local y devuelve solo el entero impreso."
            ),
            expected_answer="95",
            sandbox_files={
                "extremos_lista.py": "datos = [19, 84, 37, 62, 11]\nprint(max(datos) + min(datos))\n"
            },
            hint_tool="run_cmd",
            hint_args={"argv": ["python3", "extremos_lista.py"]},
        ),
        BenchmarkTask(
            task_id="ext_code_es_05",
            domain="code_terminal",
            subtask="file_read_inspect",
            language="es",
            prompt=(
                "Lee `nodo_local.ini` en el workspace y responde únicamente con el valor de `PUERTO_METRICS`."
            ),
            expected_answer="6180",
            sandbox_files={
                "nodo_local.ini": "[sistema]\nPUERTO_METRICS=6180\nHILOS=2\nREGION=sur\n"
            },
            hint_tool="read_file",
            hint_args={"path": "nodo_local.ini"},
        ),
        BenchmarkTask(
            task_id="ext_code_es_06",
            domain="code_terminal",
            subtask="file_read_inspect",
            language="es",
            prompt=(
                "Inspecciona `despliegue.env` en el workspace y responde solo con el número de `LOTE_MAX`."
            ),
            expected_answer="512",
            sandbox_files={
                "despliegue.env": "ENTORNO=produccion\nLOTE_MAX=512\nREINTENTOS=5\n"
            },
            hint_tool="read_file",
            hint_args={"path": "despliegue.env"},
        ),
        BenchmarkTask(
            task_id="ext_code_es_07",
            domain="code_terminal",
            subtask="file_read_inspect",
            language="es",
            prompt=(
                "Lee `estado_cola.json` en el workspace y devuelve únicamente el entero de `tareas_pendientes`."
            ),
            expected_answer="47",
            sandbox_files={
                "estado_cola.json": '{"nodo": "sur-1", "tareas_pendientes": 47, "errores": 0}\n'
            },
            hint_tool="read_file",
            hint_args={"path": "estado_cola.json"},
        ),
        BenchmarkTask(
            task_id="ext_code_es_08",
            domain="code_terminal",
            subtask="python_sandbox_exec",
            language="es",
            prompt=(
                "Ejecuta `multiplos_siete.py` en el sandbox y responde únicamente con el entero que imprime."
            ),
            expected_answer="7",
            sandbox_files={
                "multiplos_siete.py": "items = [x for x in range(1, 51) if x % 7 == 0]\nprint(len(items))\n"
            },
            hint_tool="run_cmd",
            hint_args={"argv": ["python3", "multiplos_siete.py"]},
        ),
        # =====================================================================
        # 2. math_logic (16: 8 EN + 8 ES)
        # =====================================================================
        BenchmarkTask(
            task_id="ext_math_en_01",
            domain="math_logic",
            subtask="modular_pow",
            language="en",
            prompt="Compute the exact integer value of `pow(17, 5, 97)`. Reply with only the integer.",
            expected_answer="68",
            hint_tool="calc",
            hint_args={"expr": "pow(17, 5, 97)"},
        ),
        BenchmarkTask(
            task_id="ext_math_en_02",
            domain="math_logic",
            subtask="combinatorics",
            language="en",
            prompt=(
                "Compute the exact binomial coefficient `factorial(10) // (factorial(4) * factorial(6))`. "
                "Reply with only the integer."
            ),
            expected_answer="210",
            hint_tool="calc",
            hint_args={"expr": "factorial(10) // (factorial(4) * factorial(6))"},
        ),
        BenchmarkTask(
            task_id="ext_math_en_03",
            domain="math_logic",
            subtask="floor_div_mod",
            language="en",
            prompt=(
                "Compute the exact integer value of `(98765 // 43) + (98765 % 43)`. "
                "Reply with only the integer."
            ),
            expected_answer="2333",
            hint_tool="calc",
            hint_args={"expr": "(98765 // 43) + (98765 % 43)"},
        ),
        BenchmarkTask(
            task_id="ext_math_en_04",
            domain="math_logic",
            subtask="series_sum",
            language="en",
            prompt=(
                "Compute the sum of all integers from 1 to 45 inclusive using `(45 * 46) // 2`. "
                "Reply with only the integer."
            ),
            expected_answer="1035",
            hint_tool="calc",
            hint_args={"expr": "(45 * 46) // 2"},
        ),
        BenchmarkTask(
            task_id="ext_math_en_05",
            domain="math_logic",
            subtask="lcm_calc",
            language="en",
            prompt=(
                "Compute the least common multiple of 84 and 126 via `(84 * 126) // gcd(84, 126)`. "
                "Reply with only the integer."
            ),
            expected_answer="252",
            hint_tool="calc",
            hint_args={"expr": "(84 * 126) // gcd(84, 126)"},
        ),
        BenchmarkTask(
            task_id="ext_math_en_06",
            domain="math_logic",
            subtask="quadratic",
            language="en",
            prompt=(
                "Evaluate `(3 * (19 ** 2)) - (7 * 19) + 14`. Reply with only the exact integer."
            ),
            expected_answer="964",
            hint_tool="calc",
            hint_args={"expr": "(3 * (19 ** 2)) - (7 * 19) + 14"},
        ),
        BenchmarkTask(
            task_id="ext_math_en_07",
            domain="math_logic",
            subtask="power_diff",
            language="en",
            prompt="Compute `(2 ** 13) - (3 ** 7)`. Reply with only the exact integer.",
            expected_answer="6005",
            hint_tool="calc",
            hint_args={"expr": "(2 ** 13) - (3 ** 7)"},
        ),
        BenchmarkTask(
            task_id="ext_math_en_08",
            domain="math_logic",
            subtask="weighted_sum",
            language="en",
            prompt=(
                "Compute `(18 * 27) + (34 * 15) + (12 * 41)`. Reply with only the exact integer."
            ),
            expected_answer="1488",
            hint_tool="calc",
            hint_args={"expr": "(18 * 27) + (34 * 15) + (12 * 41)"},
        ),
        BenchmarkTask(
            task_id="ext_math_es_01",
            domain="math_logic",
            subtask="modular_pow",
            language="es",
            prompt=(
                "Calcula el resto exacto `pow(23, 4, 89)`. Responde únicamente con el número entero."
            ),
            expected_answer="25",
            hint_tool="calc",
            hint_args={"expr": "pow(23, 4, 89)"},
        ),
        BenchmarkTask(
            task_id="ext_math_es_02",
            domain="math_logic",
            subtask="combinatorics",
            language="es",
            prompt=(
                "Calcula el coeficiente binomial `factorial(11) // (factorial(3) * factorial(8))`. "
                "Responde únicamente con el entero."
            ),
            expected_answer="165",
            hint_tool="calc",
            hint_args={"expr": "factorial(11) // (factorial(3) * factorial(8))"},
        ),
        BenchmarkTask(
            task_id="ext_math_es_03",
            domain="math_logic",
            subtask="floor_div_mod",
            language="es",
            prompt=(
                "Calcula el entero exacto `(75432 // 37) + (75432 % 37)`. "
                "Responde únicamente con el número entero."
            ),
            expected_answer="2064",
            hint_tool="calc",
            hint_args={"expr": "(75432 // 37) + (75432 % 37)"},
        ),
        BenchmarkTask(
            task_id="ext_math_es_04",
            domain="math_logic",
            subtask="series_sum",
            language="es",
            prompt=(
                "Calcula la suma de los enteros del 1 al 60 mediante `(60 * 61) // 2`. "
                "Responde solo con el entero."
            ),
            expected_answer="1830",
            hint_tool="calc",
            hint_args={"expr": "(60 * 61) // 2"},
        ),
        BenchmarkTask(
            task_id="ext_math_es_05",
            domain="math_logic",
            subtask="lcm_calc",
            language="es",
            prompt=(
                "Obtén el mínimo común múltiplo de 96 y 144 vía `(96 * 144) // gcd(96, 144)`. "
                "Responde únicamente con el entero."
            ),
            expected_answer="288",
            hint_tool="calc",
            hint_args={"expr": "(96 * 144) // gcd(96, 144)"},
        ),
        BenchmarkTask(
            task_id="ext_math_es_06",
            domain="math_logic",
            subtask="quadratic",
            language="es",
            prompt=(
                "Evalúa `(4 * (17 ** 2)) - (9 * 17) + 21`. Responde únicamente con el número entero."
            ),
            expected_answer="1024",
            hint_tool="calc",
            hint_args={"expr": "(4 * (17 ** 2)) - (9 * 17) + 21"},
        ),
        BenchmarkTask(
            task_id="ext_math_es_07",
            domain="math_logic",
            subtask="power_diff",
            language="es",
            prompt=(
                "Calcula `(5 ** 6) - (4 ** 5)`. Responde únicamente con el entero exacto."
            ),
            expected_answer="14601",
            hint_tool="calc",
            hint_args={"expr": "(5 ** 6) - (4 ** 5)"},
        ),
        BenchmarkTask(
            task_id="ext_math_es_08",
            domain="math_logic",
            subtask="weighted_sum",
            language="es",
            prompt=(
                "Calcula `(21 * 19) + (33 * 14) + (15 * 29)`. Responde únicamente con el entero."
            ),
            expected_answer="1296",
            hint_tool="calc",
            hint_args={"expr": "(21 * 19) + (33 * 14) + (15 * 29)"},
        ),
        # =====================================================================
        # 3. long_docs (16: 8 EN + 8 ES; 6 needle, 6 synthesis, 4 abstention)
        # =====================================================================
        BenchmarkTask(
            task_id="ext_docs_en_01",
            domain="long_docs",
            subtask="needle",
            language="en",
            prompt=(
                f"{filler_en_1}\n"
                "ARCHIVE_TOKEN_EAST: ORION-5814 (valid for Q3 verification).\n"
                f"{filler_en_2}\n"
                "Question: What is the exact ARCHIVE_TOKEN_EAST code in the document? "
                "Reply with only the code."
            ),
            expected_answer="ORION-5814",
        ),
        BenchmarkTask(
            task_id="ext_docs_en_02",
            domain="long_docs",
            subtask="needle",
            language="en",
            prompt=(
                f"{filler_en_2}\n"
                "BACKUP_CHECKPOINT_TAG: VEGA-3091 (immutable digest).\n"
                f"{filler_en_1}\n"
                "Question: What is the exact BACKUP_CHECKPOINT_TAG listed above? "
                "Reply with only the code."
            ),
            expected_answer="VEGA-3091",
        ),
        BenchmarkTask(
            task_id="ext_docs_en_03",
            domain="long_docs",
            subtask="needle",
            language="en",
            prompt=(
                f"{filler_en_1}\n"
                "DISPATCH_SEAL_ID: LYRA-8842 (sealed container).\n"
                f"{filler_en_2}\n"
                "Question: Extract the exact DISPATCH_SEAL_ID from the text. Reply with only the code."
            ),
            expected_answer="LYRA-8842",
        ),
        BenchmarkTask(
            task_id="ext_docs_en_04",
            domain="long_docs",
            subtask="synthesis",
            language="en",
            prompt=(
                "Section A (Base Quota): Cluster K9 allows a base throughput of 240 requests per second.\n"
                f"{filler_en_1}\n"
                "Section B (Peak Penalty): During backup windows, Cluster K9 throughput is reduced by 65 requests per second.\n"
                f"{filler_en_2}\n"
                "Question: What is the allowed throughput (number only) for Cluster K9 during a backup window?"
            ),
            expected_answer="175",
            hint_tool="calc",
            hint_args={"expr": "240 - 65"},
        ),
        BenchmarkTask(
            task_id="ext_docs_en_05",
            domain="long_docs",
            subtask="synthesis",
            language="en",
            prompt=(
                "Part 1 (Base Storage): Each archive blade provides 320 GB of usable space.\n"
                f"{filler_en_2}\n"
                "Part 2 (Parity Reserve): Redundant mode reserves 48 GB per blade for parity checksums.\n"
                f"{filler_en_1}\n"
                "Question: How many GB of usable space (integer only) remain per blade in redundant mode?"
            ),
            expected_answer="272",
            hint_tool="calc",
            hint_args={"expr": "320 - 48"},
        ),
        BenchmarkTask(
            task_id="ext_docs_en_06",
            domain="long_docs",
            subtask="synthesis",
            language="en",
            prompt=(
                "Clause 1 (Base Power): A standard rack draws 450 watts at idle.\n"
                f"{filler_en_1}\n"
                "Clause 2 (Accelerator Add-on): Installing the optical bridge adds 135 watts to idle draw.\n"
                f"{filler_en_2}\n"
                "Question: What is the total idle draw in watts (number only) with the optical bridge installed?"
            ),
            expected_answer="585",
            hint_tool="calc",
            hint_args={"expr": "450 + 135"},
        ),
        BenchmarkTask(
            task_id="ext_docs_en_07",
            domain="long_docs",
            subtask="abstention",
            language="en",
            is_unanswerable=True,
            prompt=(
                f"{filler_en_1}\n"
                "Note: Only passive thermal channels were logged during the night shift.\n"
                f"{filler_en_2}\n"
                "Question: What was the peak GPU voltage recorded in Bay 9? "
                "If the document does not contain enough information, reply with `INSUFFICIENT_EVIDENCE`."
            ),
            expected_answer="INSUFFICIENT_EVIDENCE",
        ),
        BenchmarkTask(
            task_id="ext_docs_en_08",
            domain="long_docs",
            subtask="abstention",
            language="en",
            is_unanswerable=True,
            prompt=(
                f"{filler_en_2}\n"
                "Summary: Quarterly filter inspections were completed on schedule.\n"
                f"{filler_en_1}\n"
                "Question: What is the vendor invoice number for the Q4 coolant shipment? "
                "If the document does not contain enough information, reply with `INSUFFICIENT_EVIDENCE`."
            ),
            expected_answer="INSUFFICIENT_EVIDENCE",
        ),
        BenchmarkTask(
            task_id="ext_docs_es_01",
            domain="long_docs",
            subtask="needle",
            language="es",
            prompt=(
                f"{filler_es_1}\n"
                "CODIGO_SELLO_NORTE: SIRIO-4179 (válido para auditoría interna).\n"
                f"{filler_es_2}\n"
                "Pregunta: ¿Cuál es exactamente el CODIGO_SELLO_NORTE del documento? "
                "Responde únicamente con el código."
            ),
            expected_answer="SIRIO-4179",
        ),
        BenchmarkTask(
            task_id="ext_docs_es_02",
            domain="long_docs",
            subtask="needle",
            language="es",
            prompt=(
                f"{filler_es_2}\n"
                "MARCA_RESPALDO_SUR: ALTAIR-6253 (registro sellado).\n"
                f"{filler_es_1}\n"
                "Pregunta: ¿Cuál es exactamente la MARCA_RESPALDO_SUR indicada en el texto? "
                "Responde solo con el código."
            ),
            expected_answer="ALTAIR-6253",
        ),
        BenchmarkTask(
            task_id="ext_docs_es_03",
            domain="long_docs",
            subtask="needle",
            language="es",
            prompt=(
                f"{filler_es_1}\n"
                "FOLIO_VERIFICACION_L2: DENEB-9317 (lote cerrado).\n"
                f"{filler_es_2}\n"
                "Pregunta: Extrae exactamente el FOLIO_VERIFICACION_L2 del documento. "
                "Responde únicamente con el código."
            ),
            expected_answer="DENEB-9317",
        ),
        BenchmarkTask(
            task_id="ext_docs_es_04",
            domain="long_docs",
            subtask="synthesis",
            language="es",
            prompt=(
                "Sección A (Capacidad Base): El nodo serie M7 admite 180 conexiones simultáneas en modo normal.\n"
                f"{filler_es_1}\n"
                "Sección B (Ajuste de Ahorro): En horario restringido, la capacidad del nodo M7 se reduce en 45 conexiones.\n"
                f"{filler_es_2}\n"
                "Pregunta: ¿Cuántas conexiones simultáneas (solo el número) admite el nodo M7 en horario restringido?"
            ),
            expected_answer="135",
            hint_tool="calc",
            hint_args={"expr": "180 - 45"},
        ),
        BenchmarkTask(
            task_id="ext_docs_es_05",
            domain="long_docs",
            subtask="synthesis",
            language="es",
            prompt=(
                "Apartado 1 (Presupuesto Base): Cada módulo de caché dispone de 512 MB asignados.\n"
                f"{filler_es_2}\n"
                "Apartado 2 (Reserva de Índice): Al activar compresión estricta se reservan 96 MB para tablas de índice.\n"
                f"{filler_es_1}\n"
                "Pregunta: ¿Cuántos MB libres (solo el entero) quedan en cada módulo con compresión estricta?"
            ),
            expected_answer="416",
            hint_tool="calc",
            hint_args={"expr": "512 - 96"},
        ),
        BenchmarkTask(
            task_id="ext_docs_es_06",
            domain="long_docs",
            subtask="synthesis",
            language="es",
            prompt=(
                "Regla 1 (Latencia Base): El enlace óptico principal tiene una latencia base de 28 ms.\n"
                f"{filler_es_1}\n"
                "Regla 2 (Salto de Cifrado): El túnel cifrado añade exactamente 19 ms adicionales.\n"
                f"{filler_es_2}\n"
                "Pregunta: ¿Cuál es la latencia total en ms (solo el número) con el túnel cifrado activo?"
            ),
            expected_answer="47",
            hint_tool="calc",
            hint_args={"expr": "28 + 19"},
        ),
        BenchmarkTask(
            task_id="ext_docs_es_07",
            domain="long_docs",
            subtask="abstention",
            language="es",
            is_unanswerable=True,
            prompt=(
                f"{filler_es_1}\n"
                "Nota: No se registraron alertas en los ventiladores auxiliares durante el turno.\n"
                f"{filler_es_2}\n"
                "Pregunta: ¿Cuál fue el consumo eléctrico máximo de la sala B4 en agosto? "
                "Si el documento no contiene información suficiente, responde `EVIDENCIA_INSUFICIENTE`."
            ),
            expected_answer="EVIDENCIA_INSUFICIENTE",
        ),
        BenchmarkTask(
            task_id="ext_docs_es_08",
            domain="long_docs",
            subtask="abstention",
            language="es",
            is_unanswerable=True,
            prompt=(
                f"{filler_es_2}\n"
                "Resumen: El inventario trimestral de cables blindados concluyó sin incidencias.\n"
                f"{filler_es_1}\n"
                "Pregunta: ¿Cuál es el número de serie del generador diésel de respaldo del sótano? "
                "Si el documento no contiene información suficiente, responde `EVIDENCIA_INSUFICIENTE`."
            ),
            expected_answer="EVIDENCIA_INSUFICIENTE",
        ),
        # =====================================================================
        # 4. factual (16: 8 EN + 8 ES; 12 L3 verifiable + 4 abstention)
        # =====================================================================
        BenchmarkTask(
            task_id="ext_fact_en_01",
            domain="factual",
            subtask="history",
            language="en",
            prompt="In what year was Magna Carta sealed at Runnymede? Reply with only the 4-digit year.",
            expected_answer="1215",
            l3_query="Magna Carta",
            hint_tool="l3_search",
            hint_args={"query": "Magna Carta", "top_k": 1},
        ),
        BenchmarkTask(
            task_id="ext_fact_en_02",
            domain="factual",
            subtask="geography",
            language="en",
            prompt="What is the summit elevation of Mount Fuji in meters? Reply with only the integer.",
            expected_answer="3776",
            l3_query="Mount Fuji",
            hint_tool="l3_search",
            hint_args={"query": "Mount Fuji", "top_k": 1},
        ),
        BenchmarkTask(
            task_id="ext_fact_en_03",
            domain="factual",
            subtask="history",
            language="en",
            prompt="In what year did the Suez Canal officially open? Reply with only the 4-digit year.",
            expected_answer="1869",
            l3_query="Suez Canal",
            hint_tool="l3_search",
            hint_args={"query": "Suez Canal", "top_k": 1},
        ),
        BenchmarkTask(
            task_id="ext_fact_en_04",
            domain="factual",
            subtask="science",
            language="en",
            prompt="What is the atomic number of Iron (Fe)? Reply with only the integer.",
            expected_answer="26",
            l3_query="Iron atomic number",
            hint_tool="l3_search",
            hint_args={"query": "Iron atomic number", "top_k": 1},
        ),
        BenchmarkTask(
            task_id="ext_fact_en_05",
            domain="factual",
            subtask="science",
            language="en",
            prompt="In what year was CERN formally established? Reply with only the 4-digit year.",
            expected_answer="1954",
            l3_query="CERN established",
            hint_tool="l3_search",
            hint_args={"query": "CERN established", "top_k": 1},
        ),
        BenchmarkTask(
            task_id="ext_fact_en_06",
            domain="factual",
            subtask="science",
            language="en",
            prompt="In what year did Alexander Fleming discover penicillin? Reply with only the 4-digit year.",
            expected_answer="1928",
            l3_query="Penicillin Fleming",
            hint_tool="l3_search",
            hint_args={"query": "Penicillin Fleming", "top_k": 1},
        ),
        BenchmarkTask(
            task_id="ext_fact_en_07",
            domain="factual",
            subtask="abstention",
            language="en",
            is_unanswerable=True,
            prompt=(
                "According to the offline L3 encyclopedia, what was the exact internal serial number "
                "of the prototype sensor deployed at Observatory Borealis-9 in 2029? "
                "If no verified evidence is available, reply with `INSUFFICIENT_EVIDENCE`."
            ),
            expected_answer="INSUFFICIENT_EVIDENCE",
            l3_query="Observatory Borealis-9 prototype sensor",
            hint_tool="l3_search",
            hint_args={"query": "Observatory Borealis-9", "top_k": 1},
        ),
        BenchmarkTask(
            task_id="ext_fact_en_08",
            domain="factual",
            subtask="abstention",
            language="en",
            is_unanswerable=True,
            prompt=(
                "According to the offline L3 encyclopedia, how many private archival crates were "
                "transferred to Vault Meridian-4 last quarter? "
                "If no verified evidence is available, reply with `INSUFFICIENT_EVIDENCE`."
            ),
            expected_answer="INSUFFICIENT_EVIDENCE",
            l3_query="Vault Meridian-4 archival crates",
            hint_tool="l3_search",
            hint_args={"query": "Vault Meridian-4", "top_k": 1},
        ),
        BenchmarkTask(
            task_id="ext_fact_es_01",
            domain="factual",
            subtask="history",
            language="es",
            prompt="¿En qué año se firmó la Paz de Westfalia? Responde únicamente con el año de 4 dígitos.",
            expected_answer="1648",
            l3_query="Paz de Westfalia",
            hint_tool="l3_search",
            hint_args={"query": "Paz de Westfalia", "top_k": 1},
        ),
        BenchmarkTask(
            task_id="ext_fact_es_02",
            domain="factual",
            subtask="geography",
            language="es",
            prompt="¿Cuál es la altitud oficial en metros del Monte Aconcagua? Responde solo con el entero.",
            expected_answer="6961",
            l3_query="Monte Aconcagua",
            hint_tool="l3_search",
            hint_args={"query": "Monte Aconcagua", "top_k": 1},
        ),
        BenchmarkTask(
            task_id="ext_fact_es_03",
            domain="factual",
            subtask="geography",
            language="es",
            prompt="¿Cuál es la altitud verificada en metros del pico K2? Responde únicamente con el entero.",
            expected_answer="8611",
            l3_query="Monte K2",
            hint_tool="l3_search",
            hint_args={"query": "Monte K2", "top_k": 1},
        ),
        BenchmarkTask(
            task_id="ext_fact_es_04",
            domain="factual",
            subtask="science",
            language="es",
            prompt="¿Cuál es el número atómico del cobre (Cu)? Responde únicamente con el número entero.",
            expected_answer="29",
            l3_query="cobre número atómico",
            hint_tool="l3_search",
            hint_args={"query": "cobre número atómico", "top_k": 1},
        ),
        BenchmarkTask(
            task_id="ext_fact_es_05",
            domain="factual",
            subtask="history",
            language="es",
            prompt=(
                "¿En qué año transmitió mensajes el primer cable telegráfico transatlántico operativo? "
                "Responde únicamente con el año de 4 dígitos."
            ),
            expected_answer="1858",
            l3_query="cable telegráfico transatlántico",
            hint_tool="l3_search",
            hint_args={"query": "cable telegráfico transatlántico", "top_k": 1},
        ),
        BenchmarkTask(
            task_id="ext_fact_es_06",
            domain="factual",
            subtask="history",
            language="es",
            prompt="¿En qué año fue promulgada la Constitución de Cádiz? Responde solo con el año de 4 dígitos.",
            expected_answer="1812",
            l3_query="Constitución de Cádiz",
            hint_tool="l3_search",
            hint_args={"query": "Constitución de Cádiz", "top_k": 1},
        ),
        BenchmarkTask(
            task_id="ext_fact_es_07",
            domain="factual",
            subtask="abstention",
            language="es",
            is_unanswerable=True,
            prompt=(
                "Según la enciclopedia offline L3, ¿cuál fue el código exacto de calibración del "
                "telescopio privado Antares-12 en 2029? "
                "Si no existe evidencia verificada, responde `EVIDENCIA_INSUFICIENTE`."
            ),
            expected_answer="EVIDENCIA_INSUFICIENTE",
            l3_query="telescopio Antares-12 calibración",
            hint_tool="l3_search",
            hint_args={"query": "telescopio Antares-12", "top_k": 1},
        ),
        BenchmarkTask(
            task_id="ext_fact_es_08",
            domain="factual",
            subtask="abstention",
            language="es",
            is_unanswerable=True,
            prompt=(
                "Según la enciclopedia offline L3, ¿cuántos paquetes confidenciales se archivaron en "
                "la bóveda Australis el trimestre pasado? "
                "Si no existe evidencia verificada, responde `EVIDENCIA_INSUFICIENTE`."
            ),
            expected_answer="EVIDENCIA_INSUFICIENTE",
            l3_query="bóveda Australis paquetes",
            hint_tool="l3_search",
            hint_args={"query": "bóveda Australis", "top_k": 1},
        ),
    ]
    return tasks


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    stripped = raw_text.strip()
    m_tool = re.search(r"<tool_call>(.*?)</tool_call>", stripped, re.DOTALL)
    if m_tool is not None:
        stripped = m_tool.group(1).strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    idx = stripped.find("{")
    while idx != -1:
        try:
            obj, _ = decoder.raw_decode(stripped, idx)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        idx = stripped.find("{", idx + 1)
    return None


def _run_single_task(
    task: BenchmarkTask,
    condition: Condition,
    backend: ModelBackend,
    l3_store: L3Encyclopedia,
    *,
    max_new_tokens: int,
    clock_fn: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    t_start = float(clock_fn())
    with tempfile.TemporaryDirectory(prefix=f"trace_4b_{task.task_id}_") as tmp_dir:
        sandbox = Sandbox(SandboxPolicy(root_dir=Path(tmp_dir), allow_write=True))
        for rel_path, content in task.sandbox_files.items():
            sandbox.write_file(rel_path, content)

        registry = TIRToolRegistry(sandbox=sandbox, l3_store=l3_store)
        prompt = task.prompt

        # C2 y C3 reciben contexto L3 offline si hay consulta factual asociada
        if condition in ("C2", "C3") and task.l3_query:
            hits = l3_store.search(task.l3_query, top_k=2)
            if hits:
                ctx_lines = [f"- [{h['title']}]: {h['content']}" for h in hits]
                prompt = "Contexto L3 (offline):\n" + "\n".join(ctx_lines) + "\n\n" + prompt

        total_json_calls = 0
        valid_json_calls = 0
        tool_result: dict[str, Any] | None = None

        if condition == "C3":
            # Presupuesto simétrico: dividimos max_new_tokens entre llamada JSON TIR y respuesta final
            tool_budget = max(16, max_new_tokens // 2)
            final_budget = max(16, max_new_tokens - tool_budget)
            tool_prompt = (
                f"{prompt}\n\n"
                f"Herramientas disponibles: {registry.available_tools()}.\n"
                'Responde con un único objeto JSON: {"tool": "<nombre>", "args": {...}} '
                'o {"tool": "none", "args": {}}.'
            )
            tool_gen = backend.generate(tool_prompt, max_new_tokens=tool_budget, json_constrained=True)
            total_json_calls += 1
            parsed_call = _extract_json_object(tool_gen.text)
            if parsed_call is not None:
                valid_json_calls += 1
                tool_name = str(parsed_call.get("tool", "none"))
                tool_args = parsed_call.get("args", {})
                if tool_name != "none" and isinstance(tool_args, dict):
                    tool_result = registry.dispatch(tool_name, tool_args)
                    prompt = (
                        f"{prompt}\n\n"
                        f"Resultado de herramienta `{tool_name}`: {json.dumps(tool_result, ensure_ascii=False)}\n"
                        "Responde ahora de forma concisa con el resultado final exacto."
                    )
            final_gen = backend.generate(prompt, max_new_tokens=final_budget, json_constrained=False)
            tokens_used = tool_gen.tokens_generated + final_gen.tokens_generated
            think_tokens = count_think_tokens(tool_gen.text) + count_think_tokens(final_gen.text)
            token_confs = list(tool_gen.token_confidences) + list(final_gen.token_confidences)
            token_corrs = list(tool_gen.token_correctness) + list(final_gen.token_correctness)
        else:
            final_gen = backend.generate(prompt, max_new_tokens=max_new_tokens, json_constrained=False)
            tokens_used = final_gen.tokens_generated
            think_tokens = count_think_tokens(final_gen.text)
            token_confs = list(final_gen.token_confidences)
            token_corrs = list(final_gen.token_correctness)

        passed = grade_answer(
            final_gen.text,
            task.expected_answer,
            is_unanswerable=task.is_unanswerable,
        )
        resp_conf = _derive_response_confidence(final_gen)
        elapsed_s = max(0.0, float(clock_fn()) - t_start)

        return {
            "task_id": task.task_id,
            "domain": task.domain,
            "subtask": task.subtask,
            "language": task.language,
            "is_unanswerable": task.is_unanswerable,
            "expected": task.expected_answer,
            "response": final_gen.text,
            "passed": passed,
            "response_confidence": resp_conf,
            "token_confidences": token_confs,
            "token_correctness": token_corrs,
            "tokens_used": tokens_used,
            "think_tokens": think_tokens,
            "elapsed_s": elapsed_s,
            "tool_result": tool_result,
            "total_json_calls": total_json_calls,
            "valid_json_calls": valid_json_calls,
            "sandbox_blocked_attempts": sandbox.blocked_count,
            "out_of_sandbox_executions": 0,
        }


def evaluate_4domains(
    backend: ModelBackend,
    *,
    conditions: tuple[Condition, ...] = ("C1", "C2", "C3"),
    tasks: list[BenchmarkTask] | None = None,
    max_new_tokens: int = 64,
    l3_corpus: list[tuple[str, str, str]] | None = None,
    clock_fn: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """Ejecuta la matriz simétrica C1-C3 sobre los 4 dominios y devuelve el reporte estructurado."""
    eval_tasks = tasks if tasks is not None else build_default_tasks()
    if l3_corpus is not None:
        corpus = l3_corpus
    elif eval_tasks and all(t.task_id.startswith("ext_") for t in eval_tasks):
        corpus = build_extended_l3_corpus()
    elif any(t.task_id.startswith("ext_") for t in eval_tasks):
        corpus = build_default_l3_corpus() + build_extended_l3_corpus()
    else:
        corpus = build_default_l3_corpus()
    clock = clock_fn if clock_fn is not None else time.monotonic

    l3_store = L3Encyclopedia(":memory:")
    l3_store.bulk_load(corpus)

    report: dict[str, Any] = {
        "max_new_tokens_budget": max_new_tokens,
        "num_tasks": len(eval_tasks),
        "out_of_sandbox_executions": 0,
        "conditions": {},
    }

    try:
        for cond in conditions:
            per_task_results: list[dict[str, Any]] = []
            all_tok_confs: list[float] = []
            all_tok_corrs: list[int] = []
            resp_probs: list[float] = []
            resp_outcomes: list[int] = []

            for task in eval_tasks:
                res = _run_single_task(
                    task,
                    cond,
                    backend,
                    l3_store,
                    max_new_tokens=max_new_tokens,
                    clock_fn=clock,
                )
                per_task_results.append(res)
                if len(res["token_confidences"]) == len(res["token_correctness"]):
                    all_tok_confs.extend(res["token_confidences"])
                    all_tok_corrs.extend(res["token_correctness"])
                resp_probs.append(float(res["response_confidence"]))
                resp_outcomes.append(1 if res["passed"] else 0)

            by_domain: dict[str, float] = {}
            for dom in ("code_terminal", "math_logic", "long_docs", "factual"):
                dom_rows = [r for r in per_task_results if r["domain"] == dom]
                by_domain[dom] = (
                    sum(1 for r in dom_rows if r["passed"]) / len(dom_rows)
                    if dom_rows
                    else 0.0
                )

            needle_rows = [r for r in per_task_results if r["domain"] == "long_docs" and r["subtask"] == "needle"]
            synth_rows = [r for r in per_task_results if r["domain"] == "long_docs" and r["subtask"] == "synthesis"]
            abst_doc_rows = [
                r for r in per_task_results if r["domain"] == "long_docs" and r["subtask"] == "abstention"
            ]
            long_docs_breakdown = {
                "needle": sum(1 for r in needle_rows if r["passed"]) / len(needle_rows) if needle_rows else 0.0,
                "synthesis": sum(1 for r in synth_rows if r["passed"]) / len(synth_rows) if synth_rows else 0.0,
            }
            if abst_doc_rows:
                long_docs_breakdown["abstention"] = (
                    sum(1 for r in abst_doc_rows if r["passed"]) / len(abst_doc_rows)
                )

            overall_em = (
                sum(resp_outcomes) / len(resp_outcomes) if resp_outcomes else 0.0
            )
            unanswerable_rows = [r for r in per_task_results if r["is_unanswerable"]]
            abstention_em = (
                float(sum(1 for r in unanswerable_rows if r["passed"]) / len(unanswerable_rows))
                if unanswerable_rows
                else None
            )
            mean_think_tokens = (
                float(sum(r["think_tokens"] for r in per_task_results) / len(per_task_results))
                if per_task_results
                else 0.0
            )
            total_wall_clock_s = float(sum(r["elapsed_s"] for r in per_task_results))
            solved_elapsed = [r["elapsed_s"] for r in per_task_results if r["passed"]]
            mean_wall_clock_to_correct_s = (
                float(sum(solved_elapsed) / len(solved_elapsed)) if solved_elapsed else None
            )

            cond_summary = {
                "overall_em": float(overall_em),
                "abstention_em": abstention_em,
                "by_domain": by_domain,
                "long_docs_breakdown": long_docs_breakdown,
                "token_ece": compute_token_ece(all_tok_confs, all_tok_corrs) if all_tok_confs else None,
                "response_brier": compute_response_brier(resp_probs, resp_outcomes),
                "mean_think_tokens": mean_think_tokens,
                "total_wall_clock_s": total_wall_clock_s,
                "mean_wall_clock_to_correct_s": mean_wall_clock_to_correct_s,
                "total_json_tool_calls": sum(r["total_json_calls"] for r in per_task_results),
                "valid_json_tool_calls": sum(r["valid_json_calls"] for r in per_task_results),
                "sandbox_blocked_attempts": sum(r["sandbox_blocked_attempts"] for r in per_task_results),
                "out_of_sandbox_executions": sum(r["out_of_sandbox_executions"] for r in per_task_results),
                "per_task": [
                    {
                        k: v
                        for k, v in r.items()
                        if k not in ("token_confidences", "token_correctness")
                    }
                    for r in per_task_results
                ],
            }
            report["conditions"][cond] = cond_summary
            report["out_of_sandbox_executions"] += cond_summary["out_of_sandbox_executions"]
    finally:
        l3_store.close()

    return report
