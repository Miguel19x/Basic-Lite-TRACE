# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Tests unitarios y adversariales de Fase 4B.1: Sandbox TIR, calculadora AST y SQLite L3."""

from __future__ import annotations

from pathlib import Path

import pytest

from trace.tir.l3_store import L3Encyclopedia
from trace.tir.sandbox import Sandbox, SandboxPolicy, SandboxViolationError
from trace.tir.tools import TIRToolRegistry, safe_eval_math


def test_sandbox_allows_local_read_write(tmp_path: Path):
    sb = Sandbox(SandboxPolicy(root_dir=tmp_path, allow_write=True))
    written = sb.write_file("sub/hello.txt", "hola trace")
    assert written == len(b"hola trace")
    assert sb.read_file("sub/hello.txt") == "hola trace"
    assert sb.list_dir("sub") == ["hello.txt"]
    assert sb.blocked_count == 0


def test_sandbox_blocks_parent_traversal(tmp_path: Path):
    sb = Sandbox(SandboxPolicy(root_dir=tmp_path))
    with pytest.raises(SandboxViolationError):
        sb.read_file("../outside.txt")
    assert sb.blocked_count == 1


def test_sandbox_blocks_absolute_path(tmp_path: Path):
    sb = Sandbox(SandboxPolicy(root_dir=tmp_path))
    with pytest.raises(SandboxViolationError):
        sb.read_file("/etc/passwd")
    assert sb.blocked_count == 1


def test_sandbox_blocks_symlink_escape(tmp_path: Path):
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    secret = outside_dir / "secret.txt"
    secret.write_text("top-secret", encoding="utf-8")

    jail = tmp_path / "jail"
    jail.mkdir()
    link = jail / "escape.txt"
    link.symlink_to(secret)

    sb = Sandbox(SandboxPolicy(root_dir=jail))
    with pytest.raises(SandboxViolationError):
        sb.read_file("escape.txt")
    assert sb.blocked_count == 1


def test_sandbox_blocks_forbidden_private_segment(tmp_path: Path):
    priv = tmp_path / "private"
    priv.mkdir()
    (priv / "notes.txt").write_text("confidential", encoding="utf-8")

    sb = Sandbox(SandboxPolicy(root_dir=tmp_path))
    assert "private" not in sb.list_dir(".")
    with pytest.raises(SandboxViolationError):
        sb.read_file("private/notes.txt")
    assert sb.blocked_count == 1


def test_sandbox_blocks_disallowed_command(tmp_path: Path):
    sb = Sandbox(SandboxPolicy(root_dir=tmp_path))
    with pytest.raises(SandboxViolationError):
        sb.run_command(["rm", "-rf", "."])
    assert sb.blocked_count == 1


def test_sandbox_blocks_command_argument_traversal(tmp_path: Path):
    sb = Sandbox(SandboxPolicy(root_dir=tmp_path))
    with pytest.raises(SandboxViolationError):
        sb.run_command(["cat", "../../etc/passwd"])
    assert sb.blocked_count == 1


def test_sandbox_executes_allowlisted_python_script(tmp_path: Path):
    sb = Sandbox(SandboxPolicy(root_dir=tmp_path))
    sb.write_file("prog.py", "print(6 * 7)\n")
    res = sb.run_command(["python3", "prog.py"])
    assert res["returncode"] == 0
    assert res["stdout"].strip() == "42"
    assert res["timed_out"] is False


def test_safe_eval_math_evaluates_exact_expression():
    assert safe_eval_math("(347 * 829) - (193 * 412) + (7 ** 4)") == 210548
    assert safe_eval_math("gcd(4620, 3570) * 19 + factorial(6)") == 4710


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('id')",
        "(lambda x: x)(1)",
        "2 ** 5000",
        "factorial(5000)",
        "open('/etc/passwd')",
    ],
)
def test_safe_eval_math_rejects_unsafe_expressions(expr: str):
    with pytest.raises(ValueError):
        safe_eval_math(expr)


def test_l3_encyclopedia_bm25_retrieves_relevant_entry():
    store = L3Encyclopedia(":memory:")
    try:
        store.bulk_load([
            ("Tratado de Tordesillas", "history", "Firmado en 1494 entre Castilla y Portugal."),
            ("Lago Baikal", "geography", "Profundidad máxima de 1642 metros en Siberia."),
        ])
        hits = store.search("Tratado de Tordesillas", top_k=1)
        assert len(hits) == 1
        assert "1494" in hits[0]["content"]
    finally:
        store.close()


def test_tir_registry_dispatches_calc(tmp_path: Path):
    sb = Sandbox(SandboxPolicy(root_dir=tmp_path))
    reg = TIRToolRegistry(sandbox=sb)
    assert reg.dispatch("calc", {"expr": "12 * 12"}) == {"ok": True, "result": 144}


def test_tir_registry_blocks_escape(tmp_path: Path):
    sb = Sandbox(SandboxPolicy(root_dir=tmp_path))
    reg = TIRToolRegistry(sandbox=sb)
    bad_res = reg.dispatch("read_file", {"path": "../outside.txt"})
    assert bad_res["ok"] is False
    assert bad_res["blocked"] is True
    assert sb.blocked_count == 1
