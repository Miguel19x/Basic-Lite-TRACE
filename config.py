# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class HybridConfig:
    # Entorno objetivo: Intel Core i3-1115G4 (2 núcleos físicos / 4 hilos lógicos, 16 GB RAM, Linux)
    # Medición empírica: 2 hilos físicos evitan contención OpenMP/AVX en tensores medianos (~1.67x más rápido que 4 hilos)
    seed: int = 42
    cpu_threads: int = 2
    device: str = "cpu"

    # Vocabulario por bytes (0..255) + 4 tokens especiales
    vocab_size: int = 260
    pad_token_id: int = 256
    mask_token_id: int = 257
    bos_token_id: int = 258
    eos_token_id: int = 259

    # Dimensiones de secuencia y bloques causales (CDLM + GQA)
    max_seq_len: int = 160           # 10 bloques de 16 bytes (soporte garantizado para entrada enriquecida de Opción A)
    block_size: int = 16             # 10 bloques de 16 bytes
    d_model: int = 192
    n_heads: int = 4
    n_kv_heads: int = 2
    d_ff_expert: int = 384

    # Principio 0: Traza supervisada compacta y Operandos Explícitos (Hito 2 / Opción A)
    use_scratchpad: bool = True      # Si True, incluye pw y pms antes de watt y ms con consistencia aditiva
    use_explicit_operands: bool = False # Opción A: proporciona operandos numéricos bw, mw, aw, ow, bms, mms, ams, oms en prompt

    # Principio 1: Capas ternarias BitLinear (W1.58A8) y Paginación BITCOS
    activation_bits: int = 8
    use_integer_inference: bool = False
    use_add_sub_inference: bool = False
    eps: float = 1e-6
    paged_bitcos_page_size: int = 4096  # Tamaño de página binaria (4 KB) para mmap / paginación de tensores
    paged_bitcos_format: str = "compact" # 'compact' (menor tráfico) o 'fast' (decodificación rápida)

    # Principio 2: Difusión discreta enmascarada + Consistencia CDLM
    min_mask_rate: float = 0.15
    max_mask_rate: float = 1.0
    diffusion_steps_per_block: int = 4
    consistency_loss_weight: float = 0.10

    # Principio 3: Recursión con codificación sinusoidal continua y cómputo adaptativo
    num_recursions: int = 2          # Profundidad base en entrenamiento (H1 evalúa R in {1,2,4,8})
    randomize_train_recursions: bool = False  # Si True, muestrea R_train en {1, 2, 3}
    nograd_recursions: int = 0       # 0 para que todos los pasos reciban gradiente limpio
    intermediate_loss_weight: float = 0.25
    halt_threshold: float = 0.90
    ponder_loss_weight: float = 0.01
    adaptive_recursion: bool = True  # Salida temprana si delta latente ||y_{r+1} - y_r|| / ||y_r|| < epsilon
    adaptive_recursion_tol: float = 0.05
    adaptive_diffusion: bool = True  # Salida temprana en difusión si el bloque está resuelto (confianza alta)
    adaptive_diffusion_conf: float = 0.85

    # Ponderación contra el techo trivial (priorizar bytes de dígitos sobre plantilla y PAD)
    digit_token_weight: float = 6.0  # Peso x6 en posiciones de dígitos ('0'..'9')
    template_token_weight: float = 3.5 # Peso suficiente para evitar fallos ortográficos (sub_w:, sub_ms:, etc.)
    pad_token_weight: float = 0.20   # Peso reducido en PAD post-EOS
    loss_weight_pw: float = 1.0      # Peso del scratchpad de potencia
    loss_weight_pms: float = 1.0     # Peso del scratchpad de latencia
    loss_weight_final: float = 3.5   # Peso reforzado en la suma aditiva final (watt y ms)
    loss_weight_eos: float = 0.20    # Peso del delimitador EOS

    # Principio 4: Modularidad Paginada y Enrutamiento Consciente de Memoria
    num_experts: int = 4
    capacity_factor: float = 2.0
    use_shared_expert: bool = True            # Experto compartido permanente estilo DeepSeek-V3
    use_qk_norm: bool = True                  # Normalización RMSNorm a Q y K estilo Qwen 2.5 / Gemma 2
    use_logit_softcapping: bool = True        # Acotación hiperbólica de logits (cap 30.0) estilo Gemma 2
    logit_cap_value: float = 30.0
    use_dynamic_remasking: bool = False       # Desenmascaramiento progresivo monotónico (sin oscilación de re-enmascaramiento)
    sparse_expert_dispatch: bool = True       # Omite físicamente plazas PAD e inactivas (gate <= min_gate_threshold)
    use_sparse_module_graph: bool = True      # Restringe transiciones entre expertos por aristas del grafo en r > 0
    graph_out_degree: int = 2                 # Aristas permitidas por nodo experto en r > 0 (auto-bucle + vecino)
    graph_capacity_decay: float = 0.75        # Amortiguación de presupuesto k en recursiones sucesivas r > 0
    min_gate_threshold: float = 1e-4          # Umbral mínimo para ejecutar un token en un nodo experto

    # =========================================================================
    # Sistema de Activación por Escalabilidad (TRACE Núcleo 870K -> Escalado 30B)
    # =========================================================================
    # A la escala actual (~870K parámetros, secuencias de 160 bytes en CPU i3-1115G4),
    # estos componentes de arquitectura 30B permanecen desactivados (False) para
    # garantizar máxima velocidad, gradientes limpios y resolución aritmética byte a byte.
    # Al escalar el modelo o la longitud de contexto, se activan cambiando su bandera.
    enable_nvme_io_simulation: bool = False  # True: activa penalización S(e) y buddy routing para SSD; False: expertos 100% residentes en RAM
    enable_sqlite_context_l3: bool = False   # True: activa persistencia SQLite (context_archive.db); False: contexto L1/L2 en RAM
    use_mla: bool = False                    # True: activa compresión KV latente para 128k tokens; False: GQA directa estándar para secuencias cortas
    use_byte_patching: bool = False          # True: agrupa P=4 bytes para secuencias masivas; False: 1:1 bytes para resolución aritmética exacta

    # Principio 4.1: Administrador de Movimiento de Pesos y Caché de Expertos (Modo 30B)
    max_resident_experts: int = 2             # Presupuesto estricto de expertos en RAM simultáneos (ej. 2 de 4) si enable_nvme_io_simulation=True
    lambda_read_cost: float = 0.35            # Penalización por costo de transferencia desde almacenamiento secundario
    mu_compute_cost: float = 0.05             # Penalización por costo de operaciones
    memory_escape_threshold: float = 0.70     # Utilidad mínima requerida para forzar carga de experto ausente de SSD
    enable_expert_prefetch: bool = True       # Anticipación del siguiente paso en ejecución asíncrona

    # Principio 5: Memoria Contextual Jerárquica y Compresión de Caché KV (Modo 30B)
    mla_latent_dim: int = 48                  # Dimensión latente d_c << 2 * n_kv_heads * head_dim (si use_mla=True)
    active_context_tokens: int = 64           # Nivel 1: Tamaño de la ventana activa en RAM (atención directa)
    compressed_segment_tokens: int = 32       # Nivel 2: Tamaño de segmento comprimido latente
    archive_context_enabled: bool = False     # Sincronizado con enable_sqlite_context_l3
    archive_storage_path: str = "context_archive.db"
    retrieval_top_k: int = 2                  # Número de fragmentos exactos recuperados hacia el nivel activo

    # Principio 6: Compresión Interna de Bytes (Byte Patching Jerárquico - Modo 30B)
    byte_patch_size: int = 4                  # 4 bytes por parche (reduce la longitud del tronco a 1/4 si use_byte_patching=True)
    patch_encoder_dim: int = 96               # Dimensión intermedia del codificador local ligero

    # Motor de ejecución con presupuesto (Lazy Logits, Compilación y Separación Dev/Experimento)
    lazy_vocab_projection: bool = True        # Proyecta al vocabulario solo los estados supervisados en train y R final en eval
    use_torch_compile: bool = False           # Compila regiones estables (expertos ternarios) si el backend C++ está activo
    dev_mode: bool = False                    # Modo rápido de desarrollo (subconjunto fijo de validación y R=2)
    val_samples_full: int = 384
    val_samples_dev: int = 64
    eval_recursions_full: Tuple[int, ...] = (1, 2, 4)
    eval_recursions_dev: Tuple[int, ...] = (2,)

    # Optimización AdamW (decaimiento desacoplado)
    batch_size: int = 16
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.03
    betas: tuple = (0.9, 0.95)
    max_steps: int = 800             # Presupuesto ampliado para consolidar aritmética
    eval_every: int = 100
    grad_clip: float = 1.0
    checkpoint_path: str = "trace_checkpoint.pt"
    bitcos_export_path: str = "trace_paged_bitcos.bin"

    def __post_init__(self) -> None:
        self.archive_context_enabled = self.enable_sqlite_context_l3

