# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available


import torch
import torch.nn as nn


def extract_abacus_ranks(
    input_ids: torch.Tensor,
    delimiter_rank: int = 15,
    max_rank: int = 14,
    mask_token_id: int | None = 257,
) -> torch.Tensor:
    """
    Calcula de forma 100% vectorizada en PyTorch los rangos decimales relativos
    (0=unidades, 1=decenas, 2=centenas, ...) para cada token numérico ASCII ('0'..'9', 48..57)
    o token de máscara numérica (mask_token_id), de derecha a izquierda, conforme a la formulación de McLeish et al. (NeurIPS 2024).

    Los tokens no numéricos (delimitadores, letras, padding) se asignan al rango 'delimiter_rank'.

    Args:
        input_ids: Tensor de enteros de dimensiones (B, L).
        delimiter_rank: Índice reservado para tokens no numéricos (por defecto 15).
        max_rank: Límite superior de rango decimal (por defecto 14, para números de hasta 15 cifras).
        mask_token_id: ID del token de máscara para inferencia causal/relleno (por defecto 257).

    Returns:
        Tensor de enteros (B, L) con el rango decimal de cada token.
    """
    is_digit = (input_ids >= 48) & (input_ids <= 57)
    if mask_token_id is not None:
        is_digit = is_digit | (input_ids == mask_token_id)

    # Invertir la dimensión temporal para procesar de menor a mayor significancia (unidades -> centenas)
    rev = is_digit.flip(dims=[1])

    # Algoritmo de suma acumulativa con reinicio por vectorización pura en GPU/CPU
    cs = rev.long().cumsum(dim=1)
    resets = torch.where(~rev, cs, torch.zeros_like(cs))
    last_reset = resets.cummax(dim=1).values
    ranks_rev = cs - last_reset - 1

    # Acotar al rango máximo y asignar rango de delimitador a no dígitos
    ranks_rev = torch.clamp(ranks_rev, 0, max_rank)
    ranks_rev = torch.where(rev, ranks_rev, torch.full_like(ranks_rev, delimiter_rank))

    # Revertir a la orientación original de la secuencia
    return ranks_rev.flip(dims=[1])


class AbacusEmbedding(nn.Module):
    """
    Capa de Abacus Positional Embedding que proyecta los rangos decimales de los números
    a vectores de dimensión d_model para dotar al Transformer de conciencia de orden de magnitud.
    """
    def __init__(
        self,
        d_model: int,
        max_ranks: int = 16,
        delimiter_rank: int = 15,
        mask_token_id: int | None = 257,
    ):
        super().__init__()
        assert max_ranks >= 2, f"AbacusEmbedding requiere max_ranks >= 2, recibido: {max_ranks}"
        assert 0 <= delimiter_rank < max_ranks, (
            f"delimiter_rank ({delimiter_rank}) debe estar en [0, {max_ranks})"
        )
        self.d_model = d_model
        self.max_ranks = max_ranks
        self.delimiter_rank = delimiter_rank
        self.mask_token_id = mask_token_id
        self.embed = nn.Embedding(max_ranks, d_model)

        # Inicialización normal con dispersión baja para actuar como sesgo posicional estable
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Calcula los vectores de incrustación de ábaco para los tokens de entrada.

        Args:
            input_ids: Tensor (B, L) con los IDs de tokens (0..259).

        Returns:
            Tensor (B, L, d_model) con las incrustaciones de ábaco.
        """
        ranks = extract_abacus_ranks(
            input_ids,
            delimiter_rank=self.delimiter_rank,
            max_rank=self.max_ranks - 2,
            mask_token_id=self.mask_token_id,
        )
        return self.embed(ranks)
