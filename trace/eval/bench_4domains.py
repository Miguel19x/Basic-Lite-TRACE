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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from trace.tir.l3_store import L3Encyclopedia
from trace.tir.sandbox import Sandbox, SandboxPolicy
from trace.tir.tools import TIRToolRegistry

Condition = Literal["C1", "C2", "C3"]
Domain = Literal["code_terminal", "math_logic", "long_docs", "factual"]


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


def grade_answer(response_text: str, expected_answer: str) -> bool:
    """
    Evalúa coincidencia exacta del valor semántico esperado sin inflar por plantilla.
    Exige que el valor esperado aparezca como token/número delimitado (no subcadena accidental).
    """
    exp = _normalize_text(expected_answer)
    if not exp:
        return False
    norm_resp = _normalize_text(response_text)
    if norm_resp == exp:
        return True
    pattern = rf"(?<![0-9a-zA-Z_]){re.escape(exp)}(?![0-9a-zA-Z_])"
    return re.search(pattern, norm_resp) is not None


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


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    stripped = raw_text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(stripped[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
    return None


def _run_single_task(
    task: BenchmarkTask,
    condition: Condition,
    backend: ModelBackend,
    l3_store: L3Encyclopedia,
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
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
            token_confs = list(tool_gen.token_confidences) + list(final_gen.token_confidences)
            token_corrs = list(tool_gen.token_correctness) + list(final_gen.token_correctness)
        else:
            final_gen = backend.generate(prompt, max_new_tokens=max_new_tokens, json_constrained=False)
            tokens_used = final_gen.tokens_generated
            token_confs = list(final_gen.token_confidences)
            token_corrs = list(final_gen.token_correctness)

        passed = grade_answer(final_gen.text, task.expected_answer)
        resp_conf = _derive_response_confidence(final_gen)

        return {
            "task_id": task.task_id,
            "domain": task.domain,
            "subtask": task.subtask,
            "expected": task.expected_answer,
            "response": final_gen.text,
            "passed": passed,
            "response_confidence": resp_conf,
            "token_confidences": token_confs,
            "token_correctness": token_corrs,
            "tokens_used": tokens_used,
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
) -> dict[str, Any]:
    """Ejecuta la matriz simétrica C1-C3 sobre los 4 dominios y devuelve el reporte estructurado."""
    eval_tasks = tasks if tasks is not None else build_default_tasks()
    corpus = l3_corpus if l3_corpus is not None else build_default_l3_corpus()

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
            long_docs_breakdown = {
                "needle": sum(1 for r in needle_rows if r["passed"]) / len(needle_rows) if needle_rows else 0.0,
                "synthesis": sum(1 for r in synth_rows if r["passed"]) / len(synth_rows) if synth_rows else 0.0,
            }

            overall_em = (
                sum(resp_outcomes) / len(resp_outcomes) if resp_outcomes else 0.0
            )
            cond_summary = {
                "overall_em": float(overall_em),
                "by_domain": by_domain,
                "long_docs_breakdown": long_docs_breakdown,
                "token_ece": compute_token_ece(all_tok_confs, all_tok_corrs),
                "response_brier": compute_response_brier(resp_probs, resp_outcomes),
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
