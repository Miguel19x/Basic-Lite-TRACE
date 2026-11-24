#!/usr/bin/env python3
# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available

"""
Pruebas unitarias para HierarchicalContextManager (Niveles L1, L2 y L3 con SQLite FTS5).
Verifica:
1. Ingesta de tokens superando la ventana activa sin NameError (raw_bytes, raw_text, keywords).
2. Integridad de telemetría sin fallos de NoneType en modo RAM.
3. Búsqueda y recuperación exacta en Nivel 3 con SQLite FTS5 y fallback.
4. Inyección a ventana activa unificando claves 'raw_text' y 'text'.
"""

import os
import tempfile
import unittest

from trace.core.context_manager import HierarchicalContextManager


class TestHierarchicalContextManager(unittest.TestCase):

    def test_ingest_tokens_and_archiving_ram_mode(self):
        """Verifica ingest_tokens() y retrieve_exact_fact() en modo RAM sin errores."""
        cm = HierarchicalContextManager(
            active_window_size=16,
            compressed_segment_size=16,
            enable_sqlite_context_l3=False,
        )
        try:
            sample_text = "nodo: sensor carga: baja red: simple perfil: eco\n"
            tokens = [ord(c) for c in sample_text]

            res = cm.ingest_tokens(tokens)
            self.assertGreater(res["archived_chunks_created"], 0)
            self.assertGreater(len(cm.compressed_segments), 0)
            self.assertLessEqual(res["active_tokens"], 16)

            # Telemetría en modo RAM (self.conn is None)
            telemetry = cm.get_telemetry()
            self.assertIn("active_tokens_in_ram", telemetry)
            self.assertIn("archived_chunks_on_disk", telemetry)
            self.assertEqual(telemetry["archived_chunks_on_disk"], len(cm.compressed_segments))

            # Recuperación exacta en modo RAM
            ram_results = cm.retrieve_exact_fact("sensor", top_k=2)
            self.assertGreater(len(ram_results), 0)
            self.assertIn("sensor", ram_results[0]["raw_text"])
            self.assertIn("text", ram_results[0])
        finally:
            cm.close()

    def test_sqlite_fts5_indexing_and_retrieval(self):
        """Verifica la indexación, recuperación exacta y saneamiento de caracteres especiales FTS5."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "test_context.db")
            cm = HierarchicalContextManager(
                active_window_size=32,
                compressed_segment_size=16,
                db_path=db_path,
                enable_sqlite_context_l3=True,
            )
            try:
                # Ingestar varios fragmentos incluyendo identificadores con guiones/dos puntos
                text_1 = "nodo: datacenter carga: pico red: ring-01 perfil: turbo\n"
                text_2 = "nodo: sensor carga: idle red: ninguna perfil: eco\n"
                text_3 = "nodo: gateway carga: saturada red: mesh perfil: secure\n"

                cm.ingest_tokens([ord(c) for c in text_1])
                cm.ingest_tokens([ord(c) for c in text_2])
                cm.ingest_tokens([ord(c) for c in text_3])

                telemetry = cm.get_telemetry()
                self.assertGreater(telemetry["archived_chunks_on_disk"], 0)

                # Búsqueda exacta por FTS5 de término específico
                results = cm.retrieve_exact_fact("datacenter", top_k=1)
                self.assertEqual(len(results), 1)
                self.assertIn("datacenter", results[0]["raw_text"])
                self.assertIn("raw_text", results[0])
                self.assertIn("text", results[0])

                # Caracteres especiales FTS5 no deben lanzar sqlite3.OperationalError
                for special_q in ('"foo"', "a-b", "a:b", "AND", "OR", "NOT", "a*", "ring-01"):
                    res_special = cm.retrieve_exact_fact(special_q, top_k=2)
                    self.assertIsInstance(res_special, list)

                # Inyección a ventana activa seguida de nueva ingesta (sin start_token_idx negativo)
                init_active = len(cm.active_token_ids)
                cm.inject_retrieved_to_active(results[0])
                self.assertGreater(len(cm.active_token_ids), init_active)

                extra_text = "nodo: edge carga: media red: star perfil: balanced\n"
                cm.ingest_tokens([ord(c) for c in extra_text])

                for seg in cm.compressed_segments:
                    self.assertGreaterEqual(seg["start_idx"], 0, f"start_idx negativo detectado: {seg}")

                cursor = cm.conn.cursor()
                cursor.execute("SELECT MIN(start_token_idx) FROM context_archive")
                min_start = cursor.fetchone()[0]
                self.assertIsNotNone(min_start)
                self.assertGreaterEqual(min_start, 0, f"start_token_idx negativo en SQLite: {min_start}")
            finally:
                cm.close()

    def test_inject_retrieved_unification(self):
        """Verifica la inserción correcta usando 'text' o 'raw_text'."""
        cm = HierarchicalContextManager(active_window_size=64, enable_sqlite_context_l3=False)
        try:
            chunk_a = {"text": "hecho_a: 12345"}
            chunk_b = {"raw_text": "hecho_b: 67890"}

            cm.inject_retrieved_to_active([chunk_a, chunk_b])
            active_str = bytes(cm.active_token_ids).decode("utf-8", errors="replace")
            self.assertIn("hecho_a: 12345", active_str)
            self.assertIn("hecho_b: 67890", active_str)
        finally:
            cm.close()


if __name__ == "__main__":
    unittest.main()
