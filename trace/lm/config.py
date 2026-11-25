# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available

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

    # Parámetros GQA / BitNet-2B-4T (compatibilidad retroactiva: valores por defecto = MLA clásico)
    attn_type: str = "mla"  # "mla" o "gqa"
    n_kv_heads: int = 0
    head_dim: int = 0
    rope_theta: float = 10000.0
    rope_interleaved: bool = True
    use_gated_ffn: bool = False

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

    @classmethod
    def preset_25m_bilingual(cls) -> "TRACELMConfig":
        """Preset bilingüe (EN + ES) de 25M parámetros con vocabulario BPE extendido (8192)."""
        return cls(
            vocab_size=8192,
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

    @classmethod
    def preset_45m_bilingual(cls) -> "TRACELMConfig":
        """Preset bilingüe profundo de 45M parámetros (EN + ES) con preservación total de capacidad."""
        return cls(
            vocab_size=8192,
            max_seq_len=4096,
            d_model=640,
            n_layers=14,
            n_heads=10,
            d_ff=1280,
            kv_lora_rank=160,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
            eps=1e-6,
            use_add_sub_inference=False,
            tie_word_embeddings=True,
            quantize_weights_enabled=True,
            weight_scale_type="mean",
        )

    @classmethod
    def preset_bitnet_2b_4t(cls) -> "TRACELMConfig":
        """Preset compatible con microsoft/bitnet-b1.58-2B-4T (GQA 20Q/5KV, ReLU^2 gated, RoPE theta=500k)."""
        return cls(
            vocab_size=128256,
            pad_token_id=128001,
            mask_token_id=128001,
            bos_token_id=128000,
            eos_token_id=128001,
            max_seq_len=4096,
            d_model=2560,
            n_layers=30,
            n_heads=20,
            d_ff=6912,
            eps=1e-5,
            kv_lora_rank=0,
            qk_nope_head_dim=0,
            qk_rope_head_dim=128,
            v_head_dim=128,
            use_add_sub_inference=False,
            tie_word_embeddings=True,
            quantize_weights_enabled=True,
            weight_scale_type="mean",
            attn_type="gqa",
            n_kv_heads=5,
            head_dim=128,
            rope_theta=500000.0,
            rope_interleaved=False,
            use_gated_ffn=True,
        )
