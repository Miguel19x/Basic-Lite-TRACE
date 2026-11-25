# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Fixtures compartidas de pytest para mantener aislamiento y 2 hilos físicos CPU."""

import pytest
import torch


@pytest.fixture(autouse=True, scope="session")
def configure_cpu_threads():
    """Fija 2 hilos físicos durante la sesión de tests y restaura el valor original al salir."""
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(prev_threads)
