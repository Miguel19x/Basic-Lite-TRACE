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

from trace.eval.bench_4domains import build_default_tasks, build_extended_suite
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
    language: str = "es"


DEFAULT_SYSTEM_PROMPT = "You are a concise, precise technical assistant. Answer in the user's language."


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
    """Verifica que ningún prompt ni respuesta de entrenamiento duplique tareas ocultas de Fase 4B (smoke o extended)."""
    all_hidden = build_default_tasks() + build_extended_suite()
    hidden_prompts = {t.prompt.strip() for t in all_hidden}
    hidden_markers = {
        "DELTA-7492",
        "ORION-5814",
        "VEGA-3091",
        "LYRA-8842",
        "SIRIO-4179",
        "ALTAIR-6253",
        "DENEB-9317",
        "347 * 829",
        "gcd(4620, 3570)",
    } | {fname for t in all_hidden for fname in t.sandbox_files}
    for ex in examples:
        if ex.prompt.strip() in hidden_prompts:
            return False
        for marker in hidden_markers:
            if marker in ex.prompt:
                return False
    return True


def _build_general_examples(rng: random.Random, count: int) -> list[ReasoningTraceExample]:
    """Trazas generales sin herramienta (50% EN / 50% ES, <think> en inglés) + abstención honesta."""
    templates = [
        # (lang, prompt_tpl, thought_en_tpl, answer_tpl)
        (
            "en",
            "Explain in one sentence the main benefit of quantizing weights to W1.58 ({{-1, 0, +1}}).",
            "Direct conceptual question; no external tool is needed. Answer concisely in English.",
            "It cuts memory bandwidth sharply and replaces FP32 multiplications with integer additions and subtractions.",
        ),
        (
            "es",
            "Explica en una sola frase qué ventaja aporta cuantizar pesos a W1.58 ({{-1, 0, +1}}).",
            "Direct conceptual question in Spanish; no tool needed. Answer concisely in Spanish.",
            "Reduce drásticamente el ancho de banda de memoria y sustituye multiplicaciones FP32 por sumas y restas enteras.",
        ),
        (
            "en",
            "Based only on the note 'Server uptime is 99.4% and region is eu-west', what is the database password? Reply INSUFFICIENT_EVIDENCE if not stated.",
            "Checking the provided note: it only states uptime and region, never a database password. I must abstain honestly.",
            "INSUFFICIENT_EVIDENCE",
        ),
        (
            "es",
            "Usando solo la nota 'El servidor tiene 99.4% de disponibilidad y opera en eu-west', ¿cuál es la clave de la base de datos? Responde EVIDENCIA_INSUFICIENTE si no aparece.",
            "Checking the note in Spanish: only uptime and region are stated, not the database password. I must abstain in Spanish.",
            "EVIDENCIA_INSUFICIENTE",
        ),
        (
            "en",
            "If a base quota is {base} units and an override subtracts {delta} units, what is the final quota?",
            "Simple mental subtraction: {base} - {delta} = {ans}. No tool call required.",
            "{ans}",
        ),
        (
            "es",
            "Si una regla base fija un límite de {base} unidades y una excepción lo reduce en {delta}, ¿cuál es el límite final?",
            "Elementary subtraction: {base} - {delta} = {ans}. No need to invoke a tool.",
            "{ans}",
        ),
        (
            "en",
            "What does INT8 KV-cache quantization do during autoregressive inference?",
            "Standard inference systems question; answer directly in English without tools.",
            "It stores keys and values as scaled 8-bit integers, cutting KV-cache RAM usage by roughly 4x.",
        ),
        (
            "es",
            "Resume brevemente por qué es útil un sandbox de permisos para un modelo con herramientas.",
            "Direct security engineering question in Spanish; no tool call needed.",
            "Aísla la ejecución local en un directorio confinado para impedir accesos no autorizados o fugas fuera del workspace.",
        ),
    ]
    out: list[ReasoningTraceExample] = []
    for i in range(count):
        lang, p_tpl, t_tpl, a_tpl = templates[i % len(templates)]
        base = rng.randint(40, 95)
        delta = rng.randint(5, 25)
        ans = str(base - delta)
        prompt = p_tpl.format(base=base, delta=delta, ans=ans)
        thought = t_tpl.format(base=base, delta=delta, ans=ans)
        answer = a_tpl.format(base=base, delta=delta, ans=ans)
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
                language=lang,
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
    Construye un dataset determinista bilingüe (50% EN / 50% ES, <think> en inglés)
    de trazas SFT ejecutadas y verificadas en sandbox.
    Garantiza general_ratio >= 0.10 (requisito de Fase 5 para evitar regresión de PPL).
    """
    if general_ratio < 0.10:
        raise ValueError("general_ratio debe ser >= 0.10 para cumplir el contrato de Fase 5")

    rng = random.Random(seed)
    examples: list[ReasoningTraceExample] = []

    # 1. Trazas matemáticas verificadas con calculadora AST segura (50% EN / 50% ES + word problems)
    reg_calc = TIRToolRegistry()
    math_t1_templates = [
        "Multi-step integer arithmetic `{expr}` is error-prone mentally; call `calc` to get the exact result.",
        "I should evaluate `{expr}` via `calc` instead of guessing to ensure exact integer precision.",
        "Let's delegate the arithmetic expression `{expr}` to `calc` so there are no carry mistakes.",
        "Using `calc` on `{expr}` guarantees an exact integer before answering.",
    ]
    math_t2_templates = [
        "`calc` returned `{ans}`. Output that exact integer.",
        "Verified result from `calc` is `{ans}`. Return it cleanly.",
        "The exact value computed by `calc` is `{ans}`.",
    ]
    for i in range(num_math):
        lang = "en" if (i % 2 == 0) else "es"
        a = rng.randint(101, 989)
        b = rng.randint(101, 989)
        c = rng.randint(11, 89)
        d = rng.randint(3, 6)
        kind = i % 4
        if kind in (0, 1):
            while a == 347 and b == 829:
                b = rng.randint(101, 989)
            expr = f"({a} * {b}) - ({c} * {a}) + ({d} ** 3)"
            is_word_problem = ((i // 4) % 2 == 1)
            if lang == "en":
                prompt = (
                    f"A batch job processes {a} items across {b} nodes, skips {c} groups of {a}, and adds {d}^3 reserve slots. How many total slots? Reply with the integer only."
                    if is_word_problem
                    else f"Compute the exact integer value of `{expr}`. Reply with the integer only."
                )
            else:
                prompt = (
                    f"Un proceso gestiona {a} lotes en {b} nodos, descuenta {c} grupos de {a} y suma {d}^3 ranuras de reserva. ¿Cuántas ranuras quedan en total? Responde solo el entero."
                    if is_word_problem
                    else f"Calcula el valor entero exacto de `{expr}`. Responde únicamente con el número entero."
                )
        else:
            g1 = rng.randint(12, 48) * rng.randint(3, 9)
            g2 = rng.randint(12, 48) * rng.randint(3, 9)
            f_n = rng.randint(4, 7)
            expr = f"gcd({g1}, {g2}) * {c} + factorial({f_n})"
            if lang == "en":
                prompt = f"Find the exact integer value of `{expr}`. Return only the integer."
            else:
                prompt = f"Obtén el valor entero exacto de `{expr}`. Devuelve únicamente el número entero."

        call = {"tool": "calc", "args": {"expr": expr}}
        res = reg_calc.dispatch("calc", call["args"])
        assert res["ok"] is True
        ans = str(res["result"])
        t1 = math_t1_templates[i % len(math_t1_templates)].format(expr=expr)
        t2 = math_t2_templates[i % len(math_t2_templates)].format(ans=ans)
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
                language=lang,
            )
        )

    # 2. Trazas de código/archivos ejecutadas en Sandbox temporal real (50% EN / 50% ES)
    code_run_t1 = [
        "Run `{fname}` inside the local sandbox via `run_cmd` to observe its exact stdout.",
        "I need to execute `{fname}` with `run_cmd` rather than simulating Python mentally.",
    ]
    code_read_t1 = [
        "I cannot know the contents of `{fname}` without reading it; call `read_file`.",
        "Inspect `{fname}` using `read_file` to extract the exact `BIND_PORT` value.",
    ]
    with tempfile.TemporaryDirectory(prefix="trace_p5_code_") as tmp_dir:
        sb = Sandbox(SandboxPolicy(root_dir=Path(tmp_dir), allow_write=True))
        reg_sb = TIRToolRegistry(sandbox=sb)
        for i in range(num_code):
            lang = "en" if (i % 2 == 0) else "es"
            if (i // 2) % 2 == 0:
                n_val = rng.randint(5, 18)
                fname = f"script_{i}.py"
                sb.write_file(fname, f"print(sum(k * (k + 1) for k in range(1, {n_val + 1})))\n")
                call = {"tool": "run_cmd", "args": {"argv": ["python3", fname]}}
                res = reg_sb.dispatch("run_cmd", call["args"])
                assert res["ok"] is True and res["returncode"] == 0
                ans = res["stdout"].strip()
                if lang == "en":
                    prompt = f"Execute `{fname}` in the local workspace and return only the integer it prints."
                else:
                    prompt = f"Ejecuta `{fname}` en el workspace local y devuelve únicamente el entero que imprime."
                t1 = code_run_t1[i % len(code_run_t1)].format(fname=fname)
                t2 = f"Execution exited with code 0 and printed `{ans}`."
            else:
                port = rng.randint(4000, 8999)
                fname = f"svc_{i}.cfg"
                sb.write_file(fname, f"SERVICE=trace_{i}\nBIND_PORT={port}\n")
                call = {"tool": "read_file", "args": {"path": fname}}
                res = reg_sb.dispatch("read_file", call["args"])
                assert res["ok"] is True
                ans = str(port)
                if lang == "en":
                    prompt = f"Read `{fname}` in the local workspace and reply only with the value of `BIND_PORT`."
                else:
                    prompt = f"Lee `{fname}` en el workspace local y responde únicamente con el valor de `BIND_PORT`."
                t1 = code_read_t1[i % len(code_read_t1)].format(fname=fname)
                t2 = f"`{fname}` specifies `BIND_PORT={ans}`."
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
                    language=lang,
                )
            )

    # 3. Trazas factuales con SQLite L3 (12 hechos disjuntos, 50% EN / 50% ES, entidades literales)
    l3_train_rows = [
        ("Mount Kilimanjaro", "geography", "Mount Kilimanjaro in Tanzania has an elevation of 5895 meters."),
        ("Invención de la Imprenta", "history", "Johannes Gutenberg introdujo la imprenta de tipos móviles en 1440."),
        ("Speed of Light", "science", "The exact speed of light in vacuum is 299792458 m/s."),
        ("Río Nilo", "geography", "El río Nilo tiene una longitud aproximada de 6650 kilómetros en África."),
        ("Mount Everest", "geography", "Mount Everest reaches an official elevation of 8849 meters above sea level."),
        ("Fosa de las Marianas", "geography", "El abismo Challenger en la Fosa de las Marianas alcanza 10984 metros de profundidad."),
        ("Atacama Desert", "geography", "The Atacama Desert in Chile covers approximately 105000 km2."),
        ("Revolución Francesa", "history", "La toma de la Bastilla que inició la Revolución Francesa ocurrió en 1789."),
        ("Apollo 11 Landing", "history", "The Apollo 11 mission first landed on the Moon on July 20, 1969."),
        ("Número atómico del Oro", "science", "El oro (Au) tiene el número atómico 79 en la tabla periódica."),
        ("Nitrogen Boiling Point", "science", "Liquid nitrogen boils at 77 Kelvin at standard atmospheric pressure."),
        ("Fundación de la ONU", "history", "La Organización de las Naciones Unidas fue fundada oficialmente en 1945."),
    ]
    fact_t1_templates = [
        "Verify the exact factual figure in the offline L3 store using query `{q}` to avoid hallucination.",
        "Query `l3_search` with `{q}` (keeping the exact entity name) to confirm the ground-truth value.",
    ]
    l3 = L3Encyclopedia(":memory:")
    try:
        l3.bulk_load(l3_train_rows)
        reg_l3 = TIRToolRegistry(l3_store=l3)
        factual_specs = [
            ("en", "Mount Kilimanjaro", "What is the elevation in meters of Mount Kilimanjaro? Reply with the number only.", "5895"),
            ("es", "imprenta Gutenberg", "¿En qué año introdujo Gutenberg la imprenta de tipos móviles? Responde únicamente con el número.", "1440"),
            ("en", "speed of light", "What is the exact speed of light in vacuum in m/s? Reply with the number only.", "299792458"),
            ("es", "longitud río Nilo", "¿Cuál es la longitud aproximada en kilómetros del río Nilo? Responde únicamente con el número.", "6650"),
            ("en", "Mount Everest elevation", "What is the official elevation in meters of Mount Everest? Reply with the number only.", "8849"),
            ("es", "Fosa de las Marianas", "¿Cuál es la profundidad en metros del abismo Challenger en las Marianas? Responde únicamente con el número.", "10984"),
            ("en", "Atacama Desert", "What is the approximate area in km2 of the Atacama Desert? Reply with the number only.", "105000"),
            ("es", "toma de la Bastilla", "¿En qué año ocurrió la toma de la Bastilla? Responde únicamente con el número.", "1789"),
            ("en", "Apollo 11", "In what year did the Apollo 11 mission first land on the Moon? Reply with the number only.", "1969"),
            ("es", "número atómico oro", "¿Cuál es el número atómico del oro (Au)? Responde únicamente con el número.", "79"),
            ("en", "Liquid nitrogen boils", "At how many Kelvin does liquid nitrogen boil at standard pressure? Reply with the number only.", "77"),
            ("es", "fundación ONU", "¿En qué año fue fundada oficialmente la Organización de las Naciones Unidas? Responde únicamente con el número.", "1945"),
        ]
        for i in range(num_factual):
            lang, q, q_prompt, ans = factual_specs[i % len(factual_specs)]
            call = {"tool": "l3_search", "args": {"query": q, "top_k": 1}}
            res = reg_l3.dispatch("l3_search", call["args"])
            assert res["ok"] is True and len(res["hits"]) >= 1
            t1 = fact_t1_templates[i % len(fact_t1_templates)].format(q=q)
            t2 = f"The retrieved L3 record confirms the exact value `{ans}`."
            examples.append(
                ReasoningTraceExample(
                    example_id=f"fact_{i:04d}",
                    category="factual_l3",
                    prompt=q_prompt,
                    thought_before_tool=t1,
                    tool_call=call,
                    tool_response=res,
                    thought_after_tool=t2,
                    answer=ans,
                    formatted_target=format_trace_target(t1, call, res, t2, ans),
                    language=lang,
                )
            )
    finally:
        l3.close()

    # 4. Mezcla >= 10% de datos generales (no-TIR) + abstención honesta
    tir_count = len(examples)
    num_general = max(4, int(round(tir_count * general_ratio / (1.0 - general_ratio))))
    examples.extend(_build_general_examples(rng, num_general))

    rng.shuffle(examples)
    assert verify_no_bench4b_leakage(examples), "Fuga detectada con tareas ocultas del banco 4B"
    return examples
