# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

from dataclasses import dataclass


@dataclass
class HybridConfig:
    # Entorno objetivo: Intel Core i3-1115G4 (2 núcleos físicos / 4 hilos lógicos, 16 GB RAM, Linux)
    # Medición empírica: 2 hilos físicos evitan contención OpenMP/AVX (~1.67x más rápido que 4 hilos)
    seed: int = 42
    cpu_threads: int = 2
    device: str = "cpu"

    # Vocabulario por bytes (0..255) + 4 tokens especiales
    vocab_size: int = 260
    pad_token_id: int = 256
    mask_token_id: int = 257
    bos_token_id: int = 258
    eos_token_id: int = 259

    # Dimensiones de secuencia y bloques causales (TRACE Causal + GQA)
    max_seq_len: int = 160
    block_size: int = 16
    d_model: int = 192
    n_heads: int = 4
    n_kv_heads: int = 2
    d_ff_expert: int = 384

    # Traza supervisada compacta
    use_scratchpad: bool = True
    use_explicit_operands: bool = False

    # Capas ternarias BitLinear (W1.58A8) y Paginación BitCOS
    activation_bits: int = 8
    use_integer_inference: bool = False
    use_add_sub_inference: bool = False
    eps: float = 1e-6
    paged_bitcos_page_size: int = 4096
    paged_bitcos_format: str = "fast"

    # Recursión y paso estático
    num_recursions: int = 2
    delta_t_val: float = 0.25
    adaptive_recursion: bool = True
    adaptive_recursion_tol: float = 0.05

    # Ponderación contra el techo trivial
    digit_token_weight: float = 6.0
    template_token_weight: float = 3.5
    pad_token_weight: float = 0.20
    loss_weight_pw: float = 1.0
    loss_weight_pms: float = 1.0
    loss_weight_final: float = 3.5
    loss_weight_eos: float = 0.20

    # Modularidad Paginada y MoE Residente
    num_experts: int = 4
    capacity_factor: float = 2.0
    use_shared_expert: bool = True
    use_qk_norm: bool = True
    use_logit_softcapping: bool = True
    logit_cap_value: float = 30.0

    # Memoria Contextual Jerárquica L1/L2/L3 (SQLite FTS5)
    enable_sqlite_context_l3: bool = False
    archive_context_enabled: bool = False
    archive_storage_path: str = "context_archive.db"
    retrieval_top_k: int = 2

    # Motor de ejecución con presupuesto
    lazy_vocab_projection: bool = True
    use_torch_compile: bool = False
    dev_mode: bool = False
    val_samples_full: int = 384
    val_samples_dev: int = 64
    eval_recursions_full: tuple[int, ...] = (1, 2, 4)
    eval_recursions_dev: tuple[int, ...] = (2,)

    # Optimización AdamW
    batch_size: int = 16
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.03
    betas: tuple = (0.9, 0.95)
    max_steps: int = 800
    eval_every: int = 100
    grad_clip: float = 1.0
    checkpoint_path: str = "trace_checkpoint.pt"
    bitcos_export_path: str = "trace_paged_bitcos.bin"

    def __post_init__(self) -> None:
        self.archive_context_enabled = self.enable_sqlite_context_l3
