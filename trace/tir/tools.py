# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Registro de herramientas TIR (calculadora AST segura, archivos/terminal en sandbox y búsqueda L3)."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from trace.tir.l3_store import L3Encyclopedia
from trace.tir.sandbox import Sandbox, SandboxViolationError

_BIN_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_SAFE_FUNCS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": math.factorial,
    "gcd": math.gcd,
}
_SAFE_CONSTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}


def _eval_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name) and node.id in _SAFE_CONSTS:
        return _SAFE_CONSTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("Exponente demasiado grande en calculadora segura")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCS:
        if node.keywords:
            raise ValueError("No se admiten argumentos nombrados en calculadora segura")
        args = [_eval_node(a) for a in node.args]
        if node.func.id == "factorial" and (len(args) != 1 or not isinstance(args[0], int) or args[0] > 1000):
            raise ValueError("factorial requiere entero <= 1000")
        return _SAFE_FUNCS[node.func.id](*args)
    raise ValueError(f"Expresión matemática no permitida: {ast.dump(node)}")


def safe_eval_math(expr: str) -> float | int:
    """Evalúa expresiones matemáticas puras vía AST sin jamás llamar a eval() ni exec()."""
    if not expr or len(expr) > 256:
        raise ValueError("Expresión vacía o demasiado larga")
    tree = ast.parse(expr.strip(), mode="eval")
    return _eval_node(tree)


class TIRToolRegistry:
    """Despachador JSON determinista para llamadas TIR bajo sandbox."""

    def __init__(self, sandbox: Sandbox | None = None, l3_store: L3Encyclopedia | None = None):
        self.sandbox = sandbox
        self.l3_store = l3_store

    def available_tools(self) -> list[str]:
        tools = ["calc"]
        if self.sandbox is not None:
            tools.extend(["read_file", "write_file", "list_dir", "run_cmd"])
        if self.l3_store is not None:
            tools.append("l3_search")
        return tools

    def dispatch(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(args, dict):
            return {"ok": False, "error": "args debe ser un objeto JSON"}
        try:
            if tool_name == "calc":
                res = safe_eval_math(str(args.get("expr", "")))
                return {"ok": True, "result": res}
            if tool_name == "l3_search":
                if self.l3_store is None:
                    return {"ok": False, "error": "L3 store no configurado"}
                hits = self.l3_store.search(str(args.get("query", "")), top_k=int(args.get("top_k", 3)))
                return {"ok": True, "hits": hits}
            if self.sandbox is None:
                return {"ok": False, "error": f"Herramienta '{tool_name}' requiere sandbox activo"}
            if tool_name == "read_file":
                return {"ok": True, "content": self.sandbox.read_file(str(args.get("path", "")))}
            if tool_name == "write_file":
                n = self.sandbox.write_file(str(args.get("path", "")), str(args.get("content", "")))
                return {"ok": True, "bytes_written": n}
            if tool_name == "list_dir":
                return {"ok": True, "entries": self.sandbox.list_dir(str(args.get("path", ".")))}
            if tool_name == "run_cmd":
                argv = args.get("argv", [])
                return {"ok": True, **self.sandbox.run_command(argv)}
            return {"ok": False, "error": f"Herramienta desconocida: {tool_name!r}"}
        except (SandboxViolationError, ValueError, FileNotFoundError, NotADirectoryError, ZeroDivisionError) as exc:
            return {"ok": False, "error": str(exc), "blocked": isinstance(exc, SandboxViolationError)}
