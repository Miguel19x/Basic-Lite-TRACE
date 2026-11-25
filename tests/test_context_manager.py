# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Pruebas unitarias de comportamiento para HierarchicalContextManager (L1/L2/L3 SQLite FTS5)."""

import pytest

from trace.core.context_manager import HierarchicalContextManager


def test_ram_mode_evicts_tokens_beyond_active_window():
    cm = HierarchicalContextManager(
        active_window_size=16,
        compressed_segment_size=16,
        enable_sqlite_context_l3=False,
    )
    try:
        sample_text = "nodo: sensor carga: baja red: simple perfil: eco\n"
        res = cm.ingest_tokens([ord(c) for c in sample_text])
        telemetry = cm.get_telemetry()

        assert res["archived_chunks_created"] > 0
        assert res["active_tokens"] <= 16
        assert telemetry["active_tokens_in_ram"] == res["active_tokens"]
        assert telemetry["archived_chunks_on_disk"] == res["archived_chunks_created"]
    finally:
        cm.close()


def test_ram_mode_retrieves_archived_fact():
    cm = HierarchicalContextManager(
        active_window_size=16,
        compressed_segment_size=16,
        enable_sqlite_context_l3=False,
    )
    try:
        sample_text = "nodo: sensor carga: baja red: simple perfil: eco\n"
        cm.ingest_tokens([ord(c) for c in sample_text])
        results = cm.retrieve_exact_fact("sensor", top_k=2)

        assert len(results) >= 1
        assert "sensor" in results[0]["raw_text"]
        assert results[0]["text"] == results[0]["raw_text"]
    finally:
        cm.close()


def test_sqlite_fts5_retrieves_exact_indexed_term(tmp_path):
    db_path = str(tmp_path / "context.db")
    cm = HierarchicalContextManager(
        active_window_size=32,
        compressed_segment_size=16,
        db_path=db_path,
        enable_sqlite_context_l3=True,
    )
    try:
        for text in (
            "nodo: datacenter carga: pico red: ring-01 perfil: turbo\n",
            "nodo: sensor carga: idle red: ninguna perfil: eco\n",
            "nodo: gateway carga: saturada red: mesh perfil: secure\n",
        ):
            cm.ingest_tokens([ord(c) for c in text])

        assert cm.get_telemetry()["archived_chunks_on_disk"] > 0
        results = cm.retrieve_exact_fact("datacenter", top_k=1)
        assert len(results) == 1
        assert "datacenter" in results[0]["raw_text"]
        assert results[0]["text"] == results[0]["raw_text"]
    finally:
        cm.close()


@pytest.mark.parametrize("special_query", ['"foo"', "a-b", "a:b", "AND", "OR", "NOT", "a*"])
def test_sqlite_fts5_sanitizes_operator_queries_without_error(tmp_path, special_query: str):
    db_path = str(tmp_path / "context_special.db")
    cm = HierarchicalContextManager(
        active_window_size=32,
        compressed_segment_size=16,
        db_path=db_path,
        enable_sqlite_context_l3=True,
    )
    try:
        cm.ingest_tokens([ord(c) for c in "nodo: datacenter carga: pico red: ring-01 perfil: turbo\n"])
        cm.ingest_tokens([ord(c) for c in "nodo: sensor carga: idle red: ninguna perfil: eco\n"])

        res_special = cm.retrieve_exact_fact(special_query, top_k=2)
        assert isinstance(res_special, list)
        assert all(isinstance(r, dict) and "raw_text" in r and "text" in r for r in res_special)
    finally:
        cm.close()


def test_sqlite_fts5_retrieves_hyphenated_identifier(tmp_path):
    db_path = str(tmp_path / "context_hyphen.db")
    cm = HierarchicalContextManager(
        active_window_size=32,
        compressed_segment_size=32,
        db_path=db_path,
        enable_sqlite_context_l3=True,
    )
    try:
        cm.ingest_tokens([ord(c) for c in "red: ring-01 perfil: turbo\n"])
        cm.ingest_tokens([ord(c) for c in "nodo: sensor red: ninguna perfil: eco carga: idle\n"])

        results = cm.retrieve_exact_fact("ring-01", top_k=2)
        assert len(results) >= 1
        assert any("ring-01" in r["raw_text"] for r in results)
    finally:
        cm.close()


def test_inject_retrieved_unifies_text_keys(tmp_path):
    db_path = str(tmp_path / "context_inject.db")
    cm = HierarchicalContextManager(
        active_window_size=64,
        compressed_segment_size=32,
        db_path=db_path,
        enable_sqlite_context_l3=True,
    )
    try:
        before = cm.get_telemetry()["active_tokens_in_ram"]
        cm.inject_retrieved_to_active([{"text": "hecho_a: 12345 "}, {"raw_text": "hecho_b: 67890 "}])
        after = cm.get_telemetry()["active_tokens_in_ram"]
        assert after > before

        # Los hechos inyectados permanecen fijados en la ventana activa incluso tras nueva ingesta
        filler = "nodo: edge carga: media red: star perfil: balanced " * 3
        cm.ingest_tokens([ord(c) for c in filler])

        active_str = bytes(cm.active_token_ids).decode("utf-8", errors="replace")
        assert "hecho_a: 12345" in active_str
        assert "hecho_b: 67890" in active_str
    finally:
        cm.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
