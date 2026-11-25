# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Enciclopedia offline L3 basada en SQLite FTS5 (BM25, 0 dependencias externas, 0 red)."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any


def _sanitize_fts_query(query: str) -> str:
    tokens = re.findall(r"[0-9A-Za-zÁÉÍÓÚáéíóúÑñÜü_]+", query)
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens[:12])


class L3Encyclopedia:
    """Almacén factual offline en SQLite FTS5 para evaluación C2/C3."""

    def __init__(self, db_path: str | Path = ":memory:"):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS entries "
            "USING fts5(title, domain, content, tokenize='unicode61')"
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def add_entry(self, title: str, domain: str, content: str) -> None:
        self.conn.execute(
            "INSERT INTO entries (title, domain, content) VALUES (?, ?, ?)",
            (title, domain, content),
        )
        self.conn.commit()

    def bulk_load(self, rows: list[tuple[str, str, str]]) -> None:
        self.conn.executemany(
            "INSERT INTO entries (title, domain, content) VALUES (?, ?, ?)",
            rows,
        )
        self.conn.commit()

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        fts_q = _sanitize_fts_query(query)
        if not fts_q or top_k <= 0:
            return []
        cur = self.conn.execute(
            "SELECT title, domain, content, bm25(entries) AS score "
            "FROM entries WHERE entries MATCH ? ORDER BY score LIMIT ?",
            (fts_q, int(top_k)),
        )
        return [
            {
                "title": row[0],
                "domain": row[1],
                "content": row[2],
                "bm25": float(row[3]),
            }
            for row in cur.fetchall()
        ]
