# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Tests TDD para la suite extendida de 4 dominios (N=64, EN/ES, 0% fuga) y telemetría."""

from __future__ import annotations

import re
from collections import Counter

import pytest

from scripts.run_bench_4b import parse_args
from trace.eval import (
    BenchmarkTask,
    GenerationResult,
    build_default_l3_corpus,
    build_default_tasks,
    build_extended_l3_corpus,
    build_extended_suite,
    count_think_tokens,
    evaluate_4domains,
    grade_answer,
)
from trace.reasoning.dataset import (
    ReasoningTraceExample,
    build_verified_tir_dataset,
    verify_no_bench4b_leakage,
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


class _FakeClock:
    """Reloj determinista inyectable para medir wall-clock sin time.sleep()."""

    def __init__(self, ticks: list[float]) -> None:
        self._iter = iter(ticks)

    def __call__(self) -> float:
        return next(self._iter)


class _ScriptedBackend:
    """Backend determinista que devuelve respuestas predefinidas por orden de llamada."""

    def __init__(self, responses: list[str]) -> None:
        self._iter = iter(responses)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        json_constrained: bool = False,
    ) -> GenerationResult:
        text = next(self._iter)
        return GenerationResult(
            text=text,
            response_confidence=0.9,
            tokens_generated=max(1, len(text.split())),
        )


def test_extended_suite_has_64_tasks() -> None:
    tasks = build_extended_suite()
    assert len(tasks) == 64


def test_extended_suite_balances_16_tasks_per_domain() -> None:
    tasks = build_extended_suite()
    counts = Counter(t.domain for t in tasks)
    assert counts == {
        "code_terminal": 16,
        "math_logic": 16,
        "long_docs": 16,
        "factual": 16,
    }


def test_extended_suite_balances_languages_per_domain() -> None:
    tasks = build_extended_suite()
    for domain in ("code_terminal", "math_logic", "long_docs", "factual"):
        lang_counts = Counter(t.language for t in tasks if t.domain == domain)
        assert lang_counts == {"en": 8, "es": 8}, f"Desbalance de idioma en {domain}: {lang_counts}"


def test_extended_suite_has_unique_ids() -> None:
    tasks = build_extended_suite()
    ids = [t.task_id for t in tasks]
    assert len(ids) == len(set(ids))


def test_extended_suite_has_unique_prompts() -> None:
    tasks = build_extended_suite()
    prompts = [_norm(t.prompt) for t in tasks]
    assert len(prompts) == len(set(prompts))


def test_extended_suite_includes_abstention_tasks() -> None:
    tasks = build_extended_suite()
    abstentions = [t for t in tasks if t.is_unanswerable]
    assert len(abstentions) >= 4
    langs = {t.language for t in abstentions}
    assert langs == {"en", "es"}


def test_extended_suite_zero_leakage_against_default_suite() -> None:
    default_tasks = build_default_tasks()
    extended_tasks = build_extended_suite()

    default_ids = {t.task_id for t in default_tasks}
    default_prompts = {_norm(t.prompt) for t in default_tasks}
    default_files = {fname for t in default_tasks for fname in t.sandbox_files}
    default_l3_titles = {_norm(row[0]) for row in build_default_l3_corpus()}

    ext_ids = {t.task_id for t in extended_tasks}
    ext_prompts = {_norm(t.prompt) for t in extended_tasks}
    ext_files = {fname for t in extended_tasks for fname in t.sandbox_files}
    ext_l3_titles = {_norm(row[0]) for row in build_extended_l3_corpus()}

    assert default_ids.isdisjoint(ext_ids)
    assert default_prompts.isdisjoint(ext_prompts)
    assert default_files.isdisjoint(ext_files)
    assert default_l3_titles.isdisjoint(ext_l3_titles)


def test_extended_suite_zero_leakage_against_phase5_generator() -> None:
    train_examples = build_verified_tir_dataset(num_math=32, num_code=16, num_factual=12, seed=42)
    extended_tasks = build_extended_suite()

    train_prompts = {_norm(ex.prompt) for ex in train_examples}
    ext_prompts = {_norm(t.prompt) for t in extended_tasks}
    assert train_prompts.isdisjoint(ext_prompts)

    train_code_files = {
        str(ex.tool_call["args"].get("path", ""))
        for ex in train_examples
        if ex.tool_call.get("tool") == "read_file"
    } | {
        str(ex.tool_call["args"]["argv"][-1])
        for ex in train_examples
        if ex.tool_call.get("tool") == "run_cmd" and ex.tool_call["args"].get("argv")
    }
    ext_files = {fname for t in extended_tasks for fname in t.sandbox_files}
    assert train_code_files.isdisjoint(ext_files)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("plain answer without tags", 0),
        ("<think></think><answer>42</answer>", 0),
        ("<think>   \n\t  </think><answer>42</answer>", 0),
        ("<think>compute six times seven</think><answer>42</answer>", 4),
        (
            "<think>step one</think><tool_call>{}</tool_call><think>verify result forty two</think><answer>42</answer>",
            6,
        ),
    ],
)
def test_count_think_tokens_counts_blocks(text: str, expected: int) -> None:
    assert count_think_tokens(text) == expected


def test_evaluate_4domains_reports_mean_think_tokens() -> None:
    tasks = [
        BenchmarkTask(
            task_id="t_1",
            domain="math_logic",
            subtask="arithmetic",
            prompt="What is 2 + 2?",
            expected_answer="4",
            language="en",
        ),
        BenchmarkTask(
            task_id="t_2",
            domain="math_logic",
            subtask="arithmetic",
            prompt="What is 3 + 3?",
            expected_answer="6",
            language="en",
        ),
    ]
    backend = _ScriptedBackend(
        [
            "<think>two plus two equals four</think><answer>4</answer>",
            "<think>add three</think><answer>6</answer>",
        ]
    )
    report = evaluate_4domains(
        backend,
        conditions=("C1",),
        tasks=tasks,
        clock_fn=_FakeClock([0.0, 1.0, 1.0, 2.0]),
    )
    c1 = report["conditions"]["C1"]
    assert c1["per_task"][0]["think_tokens"] == 5
    assert c1["per_task"][1]["think_tokens"] == 2
    assert c1["mean_think_tokens"] == pytest.approx(3.5, abs=1e-9)


def test_evaluate_4domains_reports_wall_clock_to_correct_with_fake_clock() -> None:
    tasks = [
        BenchmarkTask(
            task_id="t_ok_1",
            domain="math_logic",
            subtask="arithmetic",
            prompt="2 + 2",
            expected_answer="4",
            language="en",
        ),
        BenchmarkTask(
            task_id="t_fail",
            domain="math_logic",
            subtask="arithmetic",
            prompt="3 + 3",
            expected_answer="6",
            language="en",
        ),
        BenchmarkTask(
            task_id="t_ok_2",
            domain="math_logic",
            subtask="arithmetic",
            prompt="4 + 4",
            expected_answer="8",
            language="en",
        ),
    ]
    backend = _ScriptedBackend(["<answer>4</answer>", "<answer>99</answer>", "<answer>8</answer>"])
    # Task 1: 1.5s (solved), Task 2: 10.0s (failed), Task 3: 2.5s (solved)
    fake_clock = _FakeClock([10.0, 11.5, 20.0, 30.0, 30.0, 32.5])
    report = evaluate_4domains(
        backend,
        conditions=("C1",),
        tasks=tasks,
        clock_fn=fake_clock,
    )
    c1 = report["conditions"]["C1"]
    assert c1["per_task"][0]["elapsed_s"] == pytest.approx(1.5, abs=1e-9)
    assert c1["per_task"][1]["elapsed_s"] == pytest.approx(10.0, abs=1e-9)
    assert c1["per_task"][2]["elapsed_s"] == pytest.approx(2.5, abs=1e-9)
    assert c1["total_wall_clock_s"] == pytest.approx(14.0, abs=1e-9)
    assert c1["mean_wall_clock_to_correct_s"] == pytest.approx(2.0, abs=1e-9)


def test_evaluate_4domains_wall_clock_to_correct_is_none_when_zero_solved() -> None:
    tasks = [
        BenchmarkTask(
            task_id="t_wrong",
            domain="math_logic",
            subtask="arithmetic",
            prompt="5 + 5",
            expected_answer="10",
            language="en",
        ),
    ]
    backend = _ScriptedBackend(["<answer>999</answer>"])
    report = evaluate_4domains(
        backend,
        conditions=("C1",),
        tasks=tasks,
        clock_fn=_FakeClock([0.0, 3.25]),
    )
    c1 = report["conditions"]["C1"]
    assert c1["total_wall_clock_s"] == pytest.approx(3.25, abs=1e-9)
    assert c1["mean_wall_clock_to_correct_s"] is None


def test_run_bench_4b_cli_defaults_to_smoke_suite() -> None:
    args = parse_args(["--model", "m.bin", "--tokenizer", "t.bin"])
    assert args.suite == "smoke"


def test_run_bench_4b_cli_accepts_extended_suite() -> None:
    args = parse_args(["--model", "m.bin", "--tokenizer", "t.bin", "--suite", "extended"])
    assert args.suite == "extended"


def test_extended_suite_ground_truth_matches_sandbox_calc_l3_oracles(tmp_path) -> None:
    from trace.tir.l3_store import L3Encyclopedia
    from trace.tir.sandbox import Sandbox, SandboxPolicy
    from trace.tir.tools import TIRToolRegistry

    l3 = L3Encyclopedia(":memory:")
    l3.bulk_load(build_extended_l3_corpus())
    sb = Sandbox(SandboxPolicy(root_dir=tmp_path, allow_write=True))
    reg = TIRToolRegistry(sandbox=sb, l3_store=l3)
    try:
        for task in build_extended_suite():
            for rel, content in task.sandbox_files.items():
                sb.write_file(rel, content)
            if task.hint_tool == "calc":
                res = reg.dispatch("calc", task.hint_args)
                assert res["ok"] is True, task.task_id
                assert str(res["result"]) == task.expected_answer, task.task_id
            elif task.hint_tool == "run_cmd":
                res = reg.dispatch("run_cmd", task.hint_args)
                assert res["ok"] is True and res["returncode"] == 0, task.task_id
                assert res["stdout"].strip() == task.expected_answer, task.task_id
            elif task.hint_tool == "read_file":
                res = reg.dispatch("read_file", task.hint_args)
                assert res["ok"] is True, task.task_id
                assert task.expected_answer in res["content"], task.task_id
            elif task.hint_tool == "l3_search":
                res = reg.dispatch("l3_search", task.hint_args)
                assert res["ok"] is True, task.task_id
                if task.is_unanswerable:
                    assert res["hits"] == [], task.task_id
                else:
                    assert len(res["hits"]) >= 1, task.task_id
                    assert task.expected_answer in res["hits"][0]["content"], task.task_id
    finally:
        l3.close()


def test_unanswerable_factual_tasks_yield_zero_fts5_hits_on_extended_corpus() -> None:
    from trace.tir.l3_store import L3Encyclopedia

    l3 = L3Encyclopedia(":memory:")
    l3.bulk_load(build_extended_l3_corpus())
    try:
        unanswerable_factual = [
            t for t in build_extended_suite() if t.domain == "factual" and t.is_unanswerable
        ]
        assert len(unanswerable_factual) == 4
        for task in unanswerable_factual:
            assert l3.search(task.l3_query, top_k=2) == [], f"{task.task_id} l3_query hit"
            hint_q = str(task.hint_args.get("query", ""))
            assert l3.search(hint_q, top_k=2) == [], f"{task.task_id} hint_args hit"
    finally:
        l3.close()


def test_unanswerable_factual_tasks_yield_zero_fts5_hits_on_combined_corpus() -> None:
    from trace.tir.l3_store import L3Encyclopedia

    l3 = L3Encyclopedia(":memory:")
    l3.bulk_load(build_default_l3_corpus() + build_extended_l3_corpus())
    try:
        unanswerable_factual = [
            t for t in build_extended_suite() if t.domain == "factual" and t.is_unanswerable
        ]
        assert len(unanswerable_factual) == 4
        for task in unanswerable_factual:
            assert l3.search(task.l3_query, top_k=2) == [], f"{task.task_id} l3_query hit"
            hint_q = str(task.hint_args.get("query", ""))
            assert l3.search(hint_q, top_k=2) == [], f"{task.task_id} hint_args hit"
    finally:
        l3.close()


def test_evaluate_4domains_c2_retrieves_only_answerable_extended_facts() -> None:
    ext_by_id = {t.task_id: t for t in build_extended_suite()}
    tasks = [ext_by_id["ext_fact_en_01"], ext_by_id["ext_fact_es_08"]]
    seen_prompts: list[str] = []

    class _PromptCaptureBackend:
        def generate(
            self,
            prompt: str,
            *,
            max_new_tokens: int,
            json_constrained: bool = False,
        ) -> GenerationResult:
            seen_prompts.append(prompt)
            if "1215" in prompt:
                return GenerationResult(text="<answer>1215</answer>", response_confidence=0.9, tokens_generated=2)
            return GenerationResult(
                text="<answer>EVIDENCIA_INSUFICIENTE</answer>",
                response_confidence=0.9,
                tokens_generated=2,
            )

    report = evaluate_4domains(
        _PromptCaptureBackend(),
        conditions=("C2",),
        tasks=tasks,
        clock_fn=_FakeClock([0.0, 1.0, 1.0, 2.0]),
    )
    c2 = report["conditions"]["C2"]
    assert "Contexto L3 (offline):" in seen_prompts[0]
    assert "Contexto L3 (offline):" not in seen_prompts[1]
    assert c2["overall_em"] == pytest.approx(1.0)
    assert c2["abstention_em"] == pytest.approx(1.0)


def test_evaluate_4domains_c3_sums_think_tokens_across_turns() -> None:
    ext_by_id = {t.task_id: t for t in build_extended_suite()}
    tasks = [ext_by_id["ext_math_en_01"]]
    backend = _ScriptedBackend(
        [
            '<think>call calc tool now</think><tool_call>{"tool":"calc","args":{"expr":"pow(17, 5, 97)"}}</tool_call>',
            "<think>tool returned sixty eight</think><answer>68</answer>",
        ]
    )
    report = evaluate_4domains(
        backend,
        conditions=("C3",),
        tasks=tasks,
        clock_fn=_FakeClock([0.0, 2.0]),
    )
    c3 = report["conditions"]["C3"]
    assert c3["overall_em"] == pytest.approx(1.0)
    assert c3["per_task"][0]["think_tokens"] == 8
    assert c3["mean_think_tokens"] == pytest.approx(8.0)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<THINK>upper case reasoning</THINK><answer>42</answer>", 3),
        ("<Think>mixed case</Think><answer>42</answer>", 2),
        ("<think>truncated reasoning without closing tag", 5),
        ("<THINK>unclosed uppercase block", 3),
        ("<think>first block</think> mid <think>second unclosed block", 5),
    ],
)
def test_count_think_tokens_handles_case_insensitive_unclosed_blocks(text: str, expected: int) -> None:
    assert count_think_tokens(text) == expected


@pytest.mark.parametrize(
    ("response", "expected", "want"),
    [
        ("<answer>99</answer> revised <answer>42</answer>", "42", True),
        ("<answer>42</answer> revised <answer>99</answer>", "42", False),
        ("<ANSWER>42</ANSWER>", "42", True),
    ],
)
def test_grade_answer_extracts_last_answer_tag(response: str, expected: str, want: bool) -> None:
    assert grade_answer(response, expected) is want


@pytest.mark.parametrize(
    ("response", "expected", "want"),
    [
        ("<think>maybe 42 or 99</think> I am not sure", "42", False),
        ("<think>truncated draft 42 without closing", "42", False),
        ("<THINK>uppercase draft 42</THINK> wrong", "42", False),
        ("<think>draft 99</think> final result is 42", "42", True),
    ],
)
def test_grade_answer_strips_think_blocks_when_answer_tag_absent(
    response: str, expected: str, want: bool
) -> None:
    assert grade_answer(response, expected) is want


@pytest.mark.parametrize(
    ("response", "expected", "want"),
    [
        ("This is not INSUFFICIENT_EVIDENCE, 42", "INSUFFICIENT_EVIDENCE", False),
        ("No es EVIDENCIA_INSUFICIENTE, el valor es 10", "EVIDENCIA_INSUFICIENTE", False),
        ("INSUFFICIENT_EVIDENCE, 42", "INSUFFICIENT_EVIDENCE", False),
        ("<answer>INSUFFICIENT_EVIDENCE</answer>", "INSUFFICIENT_EVIDENCE", True),
        ("EVIDENCIA_INSUFICIENTE", "EVIDENCIA_INSUFICIENTE", True),
    ],
)
def test_grade_answer_rejects_negated_abstention_for_unanswerable_task(
    response: str, expected: str, want: bool
) -> None:
    assert grade_answer(response, expected, is_unanswerable=True) is want


def test_evaluate_4domains_reports_long_docs_abstention_breakdown() -> None:
    ext_by_id = {t.task_id: t for t in build_extended_suite()}
    tasks = [
        ext_by_id["ext_docs_en_01"],
        ext_by_id["ext_docs_en_04"],
        ext_by_id["ext_docs_en_07"],
        ext_by_id["ext_docs_es_07"],
    ]
    backend = _ScriptedBackend(
        [
            "<answer>ORION-5814</answer>",
            "<answer>175</answer>",
            "<answer>INSUFFICIENT_EVIDENCE</answer>",
            "<answer>This is not EVIDENCIA_INSUFICIENTE, 99</answer>",
        ]
    )
    report = evaluate_4domains(
        backend,
        conditions=("C1",),
        tasks=tasks,
        clock_fn=_FakeClock([0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0]),
    )
    c1 = report["conditions"]["C1"]
    assert c1["long_docs_breakdown"]["needle"] == pytest.approx(1.0)
    assert c1["long_docs_breakdown"]["synthesis"] == pytest.approx(1.0)
    assert c1["long_docs_breakdown"]["abstention"] == pytest.approx(0.5)
    assert c1["abstention_em"] == pytest.approx(0.5)
    assert [row["is_unanswerable"] for row in c1["per_task"]] == [False, False, True, True]


def test_evaluate_4domains_abstention_em_is_none_without_unanswerable_tasks() -> None:
    ext_by_id = {t.task_id: t for t in build_extended_suite()}
    backend = _ScriptedBackend(["<answer>68</answer>"])
    report = evaluate_4domains(
        backend,
        conditions=("C1",),
        tasks=[ext_by_id["ext_math_en_01"]],
        clock_fn=_FakeClock([0.0, 1.0]),
    )
    c1 = report["conditions"]["C1"]
    assert c1["abstention_em"] is None


def test_verify_no_bench4b_leakage_detects_extended_suite_prompt() -> None:
    ext_task = build_extended_suite()[0]
    leaked = ReasoningTraceExample(
        example_id="leak_0001",
        category="code_tir",
        prompt=ext_task.prompt,
        thought_before_tool="t1",
        tool_call={"tool": "none", "args": {}},
        tool_response=None,
        thought_after_tool="",
        answer=ext_task.expected_answer,
        formatted_target="<answer>3025</answer>",
    )
    assert verify_no_bench4b_leakage([leaked]) is False


def test_run_bench_4b_cli_rejects_invalid_suite() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--model", "m.bin", "--tokenizer", "t.bin", "--suite", "invalid"])


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Based on the 2 documents provided, there is INSUFFICIENT_EVIDENCE.", "INSUFFICIENT_EVIDENCE"),
        ("The document does not mention Bay 9 voltage. INSUFFICIENT_EVIDENCE", "INSUFFICIENT_EVIDENCE"),
        ("No record for Observatory Borealis-9 in 2029: INSUFFICIENT_EVIDENCE", "INSUFFICIENT_EVIDENCE"),
        ("No hay datos de Antares-12 en 2029: EVIDENCIA_INSUFICIENTE", "EVIDENCIA_INSUFICIENTE"),
    ],
)
def test_grade_answer_accepts_untagged_abstention_with_prompt_entity_numbers(
    response: str, expected: str
) -> None:
    assert grade_answer(response, expected, is_unanswerable=True) is True


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("This is not INSUFFICIENT_EVIDENCE", "INSUFFICIENT_EVIDENCE"),
        ("Do not output INSUFFICIENT_EVIDENCE", "INSUFFICIENT_EVIDENCE"),
        ("no es EVIDENCIA_INSUFICIENTE", "EVIDENCIA_INSUFICIENTE"),
    ],
)
def test_grade_answer_rejects_negated_abstention_with_intervening_word(
    response: str, expected: str
) -> None:
    assert grade_answer(response, expected, is_unanswerable=True) is False


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("This is not INSUFFICIENT_EVIDENCE, the answer is 42.", "INSUFFICIENT_EVIDENCE"),
        ("INSUFFICIENT_EVIDENCE, the answer is 42", "INSUFFICIENT_EVIDENCE"),
        ("EVIDENCIA_INSUFICIENTE, la respuesta es 42", "EVIDENCIA_INSUFICIENTE"),
        ("INSUFFICIENT_EVIDENCE, x = 42", "INSUFFICIENT_EVIDENCE"),
        ("EVIDENCIA_INSUFICIENTE, es 42", "EVIDENCIA_INSUFICIENTE"),
        ("INSUFFICIENT_EVIDENCE, is 42", "INSUFFICIENT_EVIDENCE"),
    ],
)
def test_grade_answer_rejects_contradictory_answer_alongside_abstention(
    response: str, expected: str
) -> None:
    assert grade_answer(response, expected, is_unanswerable=True) is False


def test_grade_answer_ignores_nested_answer_tag_inside_think_block() -> None:
    response = "<think>draft <answer>42</answer></think> final is 99"
    assert grade_answer(response, "42") is False
    assert grade_answer(response, "99") is True


def test_build_verified_tir_dataset_avoids_bench4b_seed_collision() -> None:
    examples = build_verified_tir_dataset(num_math=360, num_code=2, num_factual=2, seed=5276)
    assert len(examples) > 364
    assert verify_no_bench4b_leakage(examples) is True
