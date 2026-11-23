# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

from dataclasses import dataclass


@dataclass
class TRACELMConfig:
    """Configuración para TRACELanguageModel con Multi-Head Latent Attention (MLA)."""

    # Vocabulario BPE
    vocab_size: int = 4096
    pad_token_id: int = 256
    mask_token_id: int = 257
    bos_token_id: int = 258
    eos_token_id: int = 259

    # Dimensiones del Transformer (Preset compacto por defecto para desarrollo/CPU)
    max_seq_len: int = 256
    d_model: int = 192
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 384
    eps: float = 1e-6

    # Parámetros MLA (DeepSeek style)
    kv_lora_rank: int = 64
    qk_nope_head_dim: int = 32
    qk_rope_head_dim: int = 16
    v_head_dim: int = 32

    # Parámetros de precisión, inferencia y ligadura
    use_add_sub_inference: bool = False
    tie_word_embeddings: bool = True

    # Parámetros de entrenamiento BitNet y Continual QAT
    quantize_weights_enabled: bool = True
    weight_scale_type: str = "mean"

    @classmethod
    def preset_25m(cls) -> "TRACELMConfig":
        """Preset canónico de escala 25M parámetros para entrenamiento en GPU (Hito 3.6)."""
        return cls(
            vocab_size=4096,
            max_seq_len=4096,
            d_model=512,
            n_layers=12,
            n_heads=8,
            d_ff=1024,
            kv_lora_rank=128,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
            eps=1e-6,
            use_add_sub_inference=False,
            tie_word_embeddings=True,
            quantize_weights_enabled=True,
            weight_scale_type="mean",
        )
