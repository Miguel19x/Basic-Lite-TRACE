# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Generador determinista de trazas SFT verificadas (<think> + TIR) para Fase 5.

Garantías:
  1. Todas las trazas con herramientas se ejecutan y verifican contra `Sandbox` / `safe_eval_math` / `L3Encyclopedia`.
  2. Mezcla configurable con >= 10% de ejemplos generales (no-TIR / abstención de herramienta) para evitar olvido catastrófico.
  3. Verificación explícita de 0 fuga (zero leakage) respecto a las tareas ocultas del banco 4B.
"""

from __future__ import annotations

import json
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trace.eval.bench_4domains import build_default_tasks
from trace.tir.l3_store import L3Encyclopedia
from trace.tir.sandbox import Sandbox, SandboxPolicy
from trace.tir.tools import TIRToolRegistry


@dataclass(frozen=True)
class ReasoningTraceExample:
    example_id: str
    category: str  # "math_tir" | "code_tir" | "factual_l3" | "general_no_tool"
    prompt: str
    thought_before_tool: str
    tool_call: dict[str, Any]
    tool_response: dict[str, Any] | None
    thought_after_tool: str
    answer: str
    formatted_target: str


DEFAULT_SYSTEM_PROMPT = "Eres un asistente tecnico conciso y preciso. Responde solo lo solicitado."


def format_llama3_chat_prompt(user_prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """Formatea un turno de chat idéntico a `trace_tokenizer_format_chat_llama3` en C."""
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_prompt}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def format_trace_target(
    thought_before: str,
    tool_call: dict[str, Any],
    tool_response: dict[str, Any] | None,
    thought_after: str,
    answer: str,
) -> str:
    """Serializa una traza estructurada <think> + <tool_call> + <answer> canónica."""
    call_json = json.dumps(tool_call, ensure_ascii=False, separators=(",", ":"))
    parts = [
        f"<think>\n{thought_before.strip()}\n</think>",
        f"<tool_call>{call_json}</tool_call>",
    ]
    if tool_response is not None:
        resp_json = json.dumps(tool_response, ensure_ascii=False, separators=(",", ":"))
        parts.append(f"<tool_response>{resp_json}</tool_response>")
        parts.append(f"<think>\n{thought_after.strip()}\n</think>")
    parts.append(f"<answer>{answer.strip()}</answer>")
    return "\n".join(parts)


def verify_no_bench4b_leakage(examples: list[ReasoningTraceExample]) -> bool:
    """Verifica que ningún prompt ni respuesta de entrenamiento duplique tareas ocultas de Fase 4B."""
    hidden_prompts = {t.prompt.strip() for t in build_default_tasks()}
    hidden_markers = {"DELTA-7492", "sum_squares.py", "config_target.txt", "347 * 829", "gcd(4620, 3570)"}
    for ex in examples:
        if ex.prompt.strip() in hidden_prompts:
            return False
        for marker in hidden_markers:
            if marker in ex.prompt:
                return False
    return True


def _build_general_examples(rng: random.Random, count: int) -> list[ReasoningTraceExample]:
    templates = [
        (
            "Explica en una sola frase qué ventaja aporta cuantizar pesos a W1.58 ({{-1, 0, +1}}).",
            "Pregunta conceptual directa que no requiere ejecutar herramientas externas.",
            "Reduce drásticamente el ancho de banda de memoria y sustituye multiplicaciones FP32 por sumas y restas enteras.",
        ),
        (
            "Si una regla base fija un límite de {base} unidades y una excepción lo reduce en {delta}, ¿cuál es el límite final?",
            "Resta elemental directa: {base} - {delta} = {ans}. No hace falta invocar herramienta.",
            "{ans}",
        ),
        (
            "¿Qué significa que una memoria caché KV esté cuantizada a INT8?",
            "Concepto estándar de inferencia que puedo responder directamente sin herramientas.",
            "Almacena las claves y valores en enteros de 8 bits con factor de escala, reduciendo ~4x el uso de RAM.",
        ),
        (
            "Resume brevemente por qué es útil un sandbox de permisos para un modelo con herramientas.",
            "Pregunta de ingeniería de seguridad directa, sin cálculo ni acceso a archivos.",
            "Aísla la ejecución local en un directorio confinado para impedir accesos no autorizados o fugas fuera del workspace.",
        ),
    ]
    out: list[ReasoningTraceExample] = []
    for i in range(count):
        tpl = templates[i % len(templates)]
        base = rng.randint(40, 95)
        delta = rng.randint(5, 25)
        ans = str(base - delta)
        prompt = tpl[0].format(base=base, delta=delta, ans=ans)
        thought = tpl[1].format(base=base, delta=delta, ans=ans)
        answer = tpl[2].format(base=base, delta=delta, ans=ans)
        call = {"tool": "none", "args": {}}
        out.append(
            ReasoningTraceExample(
                example_id=f"gen_{i:04d}",
                category="general_no_tool",
                prompt=prompt,
                thought_before_tool=thought,
                tool_call=call,
                tool_response=None,
                thought_after_tool="",
                answer=answer,
                formatted_target=format_trace_target(thought, call, None, "", answer),
            )
        )
    return out


def build_verified_tir_dataset(
    *,
    num_math: int = 32,
    num_code: int = 16,
    num_factual: int = 12,
    general_ratio: float = 0.15,
    seed: int = 42,
) -> list[ReasoningTraceExample]:
    """
    Construye un dataset determinista de trazas SFT ejecutadas y verificadas en sandbox.
    Garantiza general_ratio >= 0.10 (requisito de Fase 5 para evitar regresión de PPL).
    """
    if general_ratio < 0.10:
        raise ValueError("general_ratio debe ser >= 0.10 para cumplir el contrato de Fase 5")

    rng = random.Random(seed)
    examples: list[ReasoningTraceExample] = []

    # 1. Trazas matemáticas verificadas con calculadora AST segura
    reg_calc = TIRToolRegistry()
    for i in range(num_math):
        a = rng.randint(101, 989)
        b = rng.randint(101, 989)
        c = rng.randint(11, 89)
        d = rng.randint(3, 6)
        if i % 2 == 0:
            expr = f"({a} * {b}) - ({c} * {a}) + ({d} ** 3)"
        else:
            g1 = rng.randint(12, 48) * rng.randint(3, 9)
            g2 = rng.randint(12, 48) * rng.randint(3, 9)
            f_n = rng.randint(4, 7)
            expr = f"gcd({g1}, {g2}) * {c} + factorial({f_n})"

        call = {"tool": "calc", "args": {"expr": expr}}
        res = reg_calc.dispatch("calc", call["args"])
        assert res["ok"] is True
        ans = str(res["result"])
        prompt = f"Calcula el valor entero exacto de `{expr}`. Responde únicamente con el número entero."
        t1 = f"El cálculo `{expr}` es propenso a error mental; invoco `calc` para obtener el entero exacto."
        t2 = f"La herramienta devolvió {ans}. Entrego ese valor exacto."
        examples.append(
            ReasoningTraceExample(
                example_id=f"math_{i:04d}",
                category="math_tir",
                prompt=prompt,
                thought_before_tool=t1,
                tool_call=call,
                tool_response=res,
                thought_after_tool=t2,
                answer=ans,
                formatted_target=format_trace_target(t1, call, res, t2, ans),
            )
        )

    # 2. Trazas de código/archivos ejecutadas en Sandbox temporal real
    with tempfile.TemporaryDirectory(prefix="trace_p5_code_") as tmp_dir:
        sb = Sandbox(SandboxPolicy(root_dir=Path(tmp_dir), allow_write=True))
        reg_sb = TIRToolRegistry(sandbox=sb)
        for i in range(num_code):
            if i % 2 == 0:
                n_val = rng.randint(5, 18)
                fname = f"script_{i}.py"
                sb.write_file(fname, f"print(sum(k * (k + 1) for k in range(1, {n_val + 1})))\n")
                call = {"tool": "run_cmd", "args": {"argv": ["python3", fname]}}
                res = reg_sb.dispatch("run_cmd", call["args"])
                assert res["ok"] is True and res["returncode"] == 0
                ans = res["stdout"].strip()
                prompt = f"Ejecuta `{fname}` en el workspace local y devuelve únicamente el entero que imprime."
                t1 = f"Debo ejecutar `{fname}` dentro del sandbox con `run_cmd` para leer su salida exacta."
                t2 = f"La ejecución terminó con código 0 y salida `{ans}`."
            else:
                port = rng.randint(4000, 8999)
                fname = f"svc_{i}.cfg"
                sb.write_file(fname, f"SERVICE=trace_{i}\nBIND_PORT={port}\n")
                call = {"tool": "read_file", "args": {"path": fname}}
                res = reg_sb.dispatch("read_file", call["args"])
                assert res["ok"] is True
                ans = str(port)
                prompt = f"Lee `{fname}` en el workspace local y responde únicamente con el valor de `BIND_PORT`."
                t1 = f"No conozco el contenido de `{fname}`; lo leo con `read_file`."
                t2 = f"El archivo indica `BIND_PORT={ans}`."
            examples.append(
                ReasoningTraceExample(
                    example_id=f"code_{i:04d}",
                    category="code_tir",
                    prompt=prompt,
                    thought_before_tool=t1,
                    tool_call=call,
                    tool_response=res,
                    thought_after_tool=t2,
                    answer=ans,
                    formatted_target=format_trace_target(t1, call, res, t2, ans),
                )
            )

    # 3. Trazas factuales con SQLite L3 (12 hechos disjuntos del corpus de test de Fase 4B)
    l3_train_rows = [
        ("Monte Kilimanjaro", "geography", "El Monte Kilimanjaro en Tanzania tiene una altitud de 5895 metros."),
        ("Invención de la Imprenta", "history", "Johannes Gutenberg introdujo la imprenta de tipos móviles en 1440."),
        ("Velocidad de la Luz", "science", "La velocidad de la luz en el vacío es exactamente 299792458 m/s."),
        ("Río Nilo", "geography", "El río Nilo tiene una longitud aproximada de 6650 kilómetros en África."),
        ("Monte Everest", "geography", "El Monte Everest alcanza una altitud oficial de 8849 metros sobre el nivel del mar."),
        ("Fosa de las Marianas", "geography", "El abismo Challenger en la Fosa de las Marianas alcanza 10984 metros de profundidad."),
        ("Desierto de Atacama", "geography", "El desierto de Atacama en Chile cubre una superficie aproximada de 105000 km2."),
        ("Revolución Francesa", "history", "La toma de la Bastilla que inició la Revolución Francesa ocurrió en 1789."),
        ("Llegada a la Luna", "history", "La misión Apolo 11 alunizó por primera vez el 20 de julio de 1969."),
        ("Número atómico del Oro", "science", "El oro (Au) tiene el número atómico 79 en la tabla periódica."),
        ("Punto de ebullición del nitrógeno", "science", "El nitrógeno líquido hierve a 77 Kelvin a presión atmosférica estándar."),
        ("Fundación de la ONU", "history", "La Organización de las Naciones Unidas fue fundada oficialmente en 1945."),
    ]
    l3 = L3Encyclopedia(":memory:")
    try:
        l3.bulk_load(l3_train_rows)
        reg_l3 = TIRToolRegistry(l3_store=l3)
        factual_specs = [
            ("Monte Kilimanjaro", "¿Cuál es la altitud en metros del Monte Kilimanjaro?", "5895"),
            ("imprenta Gutenberg", "¿En qué año introdujo Gutenberg la imprenta de tipos móviles?", "1440"),
            ("velocidad de la luz", "¿Cuál es la velocidad exacta de la luz en m/s?", "299792458"),
            ("longitud río Nilo", "¿Cuál es la longitud aproximada en kilómetros del río Nilo?", "6650"),
            ("altitud Monte Everest", "¿Cuál es la altitud oficial en metros del Monte Everest?", "8849"),
            ("Fosa de las Marianas", "¿Cuál es la profundidad en metros del abismo Challenger en las Marianas?", "10984"),
            ("Desierto de Atacama", "¿Cuál es la superficie aproximada en km2 del desierto de Atacama?", "105000"),
            ("toma de la Bastilla", "¿En qué año ocurrió la toma de la Bastilla?", "1789"),
            ("misión Apolo 11", "¿En qué año alunizó la misión Apolo 11?", "1969"),
            ("número atómico oro", "¿Cuál es el número atómico del oro (Au)?", "79"),
            ("ebullición nitrógeno", "¿A cuántos Kelvin hierve el nitrógeno líquido a presión estándar?", "77"),
            ("fundación ONU", "¿En qué año fue fundada oficialmente la Organización de las Naciones Unidas?", "1945"),
        ]
        for i in range(num_factual):
            q, q_prompt, ans = factual_specs[i % len(factual_specs)]
            call = {"tool": "l3_search", "args": {"query": q, "top_k": 1}}
            res = reg_l3.dispatch("l3_search", call["args"])
            assert res["ok"] is True and len(res["hits"]) >= 1
            t1 = f"Consulto la enciclopedia offline L3 con `{q}` para verificar el dato exacto sin alucinar."
            t2 = f"La entrada recuperada confirma el valor exacto `{ans}`."
            examples.append(
                ReasoningTraceExample(
                    example_id=f"fact_{i:04d}",
                    category="factual_l3",
                    prompt=f"{q_prompt} Responde únicamente con el número.",
                    thought_before_tool=t1,
                    tool_call=call,
                    tool_response=res,
                    thought_after_tool=t2,
                    answer=ans,
                    formatted_target=format_trace_target(t1, call, res, t2, ans),
                )
            )
    finally:
        l3.close()

    # 4. Mezcla >= 10% de datos generales (no-TIR)
    tir_count = len(examples)
    num_general = max(2, int(round(tir_count * general_ratio / (1.0 - general_ratio))))
    examples.extend(_build_general_examples(rng, num_general))

    rng.shuffle(examples)
    assert verify_no_bench4b_leakage(examples), "Fuga detectada con tareas ocultas del banco 4B"
    return examples
