# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Sandbox estricto de permisos para ejecución local segura de herramientas TIR (0 escapes)."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_ALLOWED_COMMANDS: frozenset[str] = frozenset(
    {"python3", "gcc", "ls", "cat", "wc", "head", "tail", "grep"}
)
FORBIDDEN_PATH_PARTS: frozenset[str] = frozenset({"private", ".git", ".ssh", ".env"})


class SandboxViolationError(PermissionError):
    """Se lanza cuando una llamada TIR intenta salir del sandbox o infringir la política."""


@dataclass
class SandboxPolicy:
    root_dir: Path
    allow_write: bool = True
    allowed_commands: frozenset[str] = field(default_factory=lambda: DEFAULT_ALLOWED_COMMANDS)
    timeout_sec: float = 5.0
    max_output_bytes: int = 32768
    max_file_bytes: int = 1024 * 1024


class Sandbox:
    """Entorno confinado con auditoría completa de accesos permitidos y bloqueados."""

    def __init__(self, policy: SandboxPolicy):
        self.root = Path(policy.root_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.policy = policy
        self.audit_log: list[dict[str, Any]] = []
        self.blocked_count: int = 0

    def _record(self, action: str, target: str, allowed: bool, reason: str = "") -> None:
        self.audit_log.append({
            "action": action,
            "target": target,
            "allowed": allowed,
            "reason": reason,
        })
        if not allowed:
            self.blocked_count += 1

    def resolve_path(self, rel_or_path: str, *, for_write: bool = False) -> Path:
        if not rel_or_path or "\x00" in rel_or_path:
            self._record("path", repr(rel_or_path), False, "ruta vacía o con byte nulo")
            raise SandboxViolationError("Ruta inválida")

        raw = Path(rel_or_path)
        if raw.is_absolute():
            self._record("path", rel_or_path, False, "ruta absoluta prohibida")
            raise SandboxViolationError(f"Ruta absoluta prohibida fuera del sandbox: {rel_or_path}")

        candidate = (self.root / raw).resolve()
        if not candidate.is_relative_to(self.root):
            self._record("path", rel_or_path, False, "escape de raíz (traversal/symlink)")
            raise SandboxViolationError(f"Intento de escape del sandbox: {rel_or_path}")

        rel_parts = candidate.relative_to(self.root).parts
        for part in rel_parts:
            if part in FORBIDDEN_PATH_PARTS:
                self._record("path", rel_or_path, False, f"segmento restringido '{part}'")
                raise SandboxViolationError(f"Acceso prohibido a ruta restringida: {rel_or_path}")

        if for_write and not self.policy.allow_write:
            self._record("write", rel_or_path, False, "escritura deshabilitada por política")
            raise SandboxViolationError("Escritura deshabilitada en este sandbox")

        return candidate

    def read_file(self, rel_path: str) -> str:
        target = self.resolve_path(rel_path, for_write=False)
        if not target.is_file():
            self._record("read_file", rel_path, False, "archivo no existe")
            raise FileNotFoundError(f"Archivo no encontrado en sandbox: {rel_path}")
        size = target.stat().st_size
        if size > self.policy.max_file_bytes:
            self._record("read_file", rel_path, False, "excede max_file_bytes")
            raise SandboxViolationError(f"Archivo demasiado grande ({size} bytes)")
        self._record("read_file", rel_path, True)
        return target.read_text(encoding="utf-8", errors="replace")

    def write_file(self, rel_path: str, content: str) -> int:
        target = self.resolve_path(rel_path, for_write=True)
        encoded = content.encode("utf-8")
        if len(encoded) > self.policy.max_file_bytes:
            self._record("write_file", rel_path, False, "contenido excede max_file_bytes")
            raise SandboxViolationError("Contenido demasiado grande para escribir en sandbox")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(encoded)
        self._record("write_file", rel_path, True)
        return len(encoded)

    def list_dir(self, rel_path: str = ".") -> list[str]:
        target = self.resolve_path(rel_path, for_write=False)
        if not target.is_dir():
            self._record("list_dir", rel_path, False, "no es directorio")
            raise NotADirectoryError(f"No es directorio en sandbox: {rel_path}")
        self._record("list_dir", rel_path, True)
        return sorted(p.name for p in target.iterdir() if p.name not in FORBIDDEN_PATH_PARTS)

    def run_command(self, argv: list[str]) -> dict[str, Any]:
        if not argv or not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
            self._record("run_cmd", str(argv), False, "argv vacío o inválido")
            raise SandboxViolationError("argv debe ser una lista no vacía de cadenas")

        cmd = argv[0]
        if "/" in cmd or "\\" in cmd or cmd not in self.policy.allowed_commands:
            self._record("run_cmd", " ".join(argv), False, f"comando '{cmd}' fuera de allowlist")
            raise SandboxViolationError(f"Comando no permitido en sandbox: {cmd!r}")

        # Bloquear argumentos con rutas absolutas fuera del sandbox o traversal '..'
        for arg in argv[1:]:
            if "\x00" in arg:
                self._record("run_cmd", " ".join(argv), False, "argumento con byte nulo")
                raise SandboxViolationError("Argumento inválido con byte nulo")
            if arg.startswith("/") or ".." in Path(arg).parts:
                self._record("run_cmd", " ".join(argv), False, f"ruta sospechosa en argumento '{arg}'")
                raise SandboxViolationError(f"Argumento fuera del sandbox: {arg!r}")

        env = {
            "PATH": "/usr/bin:/bin",
            "OMP_NUM_THREADS": "2",
            "HOME": str(self.root),
            "TMPDIR": str(self.root),
            "LC_ALL": "C.UTF-8",
        }
        self._record("run_cmd", " ".join(argv), True)
        try:
            proc = subprocess.run(
                argv,
                cwd=str(self.root),
                env=env,
                shell=False,
                capture_output=True,
                text=True,
                timeout=self.policy.timeout_sec,
            )
            stdout = proc.stdout[: self.policy.max_output_bytes]
            stderr = proc.stderr[: self.policy.max_output_bytes]
            return {
                "returncode": int(proc.returncode),
                "stdout": stdout,
                "stderr": stderr,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "returncode": -1,
                "stdout": (exc.stdout or "")[: self.policy.max_output_bytes] if isinstance(exc.stdout, str) else "",
                "stderr": "timeout",
                "timed_out": True,
            }
