# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from trace.layers.bitlinear import BitLinear
from trace.layers.norm import RMSNorm
from trace.layers.rotary import RotaryEmbedding
from trace.modules.attention import build_block_causal_mask


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA estilo DeepSeek-V2/V3) con cuantización ternaria:
    - Comprime el estado de claves y valores (KV) en un espacio latente de baja dimensión:
      c_t^{KV} = W_{DKV} x_t,  donde dim(c_t^{KV}) = d_c << 2 * n_heads * head_dim.
    - Reduce la huella de memoria de la caché KV activa de 4x a 8x.
    - En atención, proyecta c_t^{KV} hacia las cabezas de atención (W_{UK}, W_{UV}).
    - Soporte para RoPE desacoplado y máscara block-causal inmutable.
    """

    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads

        # Si se utiliza byte patching, la atención opera sobre parches: ajustar block_size a parches
        patch_sz = getattr(cfg, "byte_patch_size", 1) if getattr(cfg, "use_byte_patching", True) else 1
        self.block_size = max(1, cfg.block_size // patch_sz)

        # Dimensión de compresión latente de KV (MLA)
        self.d_c = getattr(cfg, "mla_latent_dim", max(16, cfg.d_model // 4))

        # Proyecciones Q
        self.q_proj = BitLinear(cfg.d_model, cfg.d_model, eps=cfg.eps)

        # Proyección de compresión descendente (Down-projection) de KV hacia espacio latente c_t^{KV}
        self.kv_down_proj = BitLinear(cfg.d_model, self.d_c, eps=cfg.eps)
        self.kv_norm = RMSNorm(self.d_c, eps=cfg.eps)

        # Proyecciones ascendentes (Up-projections) desde espacio latente a K y V
        self.k_up_proj = nn.Linear(self.d_c, cfg.d_model, bias=False)
        self.v_up_proj = nn.Linear(self.d_c, cfg.d_model, bias=False)

        # Proyección de salida
        self.o_proj = BitLinear(cfg.d_model, cfg.d_model, eps=cfg.eps)

        # Normalizaciones QK
        self.use_qk_norm = getattr(cfg, "use_qk_norm", True)
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=cfg.eps)
            self.k_norm = RMSNorm(self.head_dim, eps=cfg.eps)

        self.rope = RotaryEmbedding(self.head_dim, cfg.max_seq_len)
        self._mask_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}

    def _get_block_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (seq_len, device)
        if key not in self._mask_cache:
            self._mask_cache[key] = build_block_causal_mask(seq_len, self.block_size, device)
        return self._mask_cache[key]

    def forward(
        self,
        x: torch.Tensor,
        latent_kv_cache: Optional[torch.Tensor] = None,  # [B, L_past, d_c]
        start_pos: int = 0,
        record_telemetry: bool = False,
        integer_mode: bool = False,
        add_sub_mode: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcula la atención retornando la salida y la caché KV comprimida en dimensión d_c.
        """
        B, L, D = x.shape

        # 1. Proyección de Consultas Q
        q = self.q_proj(x, record_telemetry, integer_mode, add_sub_mode)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        if self.use_qk_norm:
            q = self.q_norm(q)
        q = self.rope.apply_rope(q, start_pos=start_pos)

        # 2. Compresión de Claves/Valores hacia el vector latente compacto c_t^{KV}
        c_kv = self.kv_down_proj(x, record_telemetry, integer_mode, add_sub_mode)
        c_kv = self.kv_norm(c_kv)  # [B, L, d_c]

        # 3. Concatenación de caché en el espacio latente comprimido
        if latent_kv_cache is not None:
            c_kv_full = torch.cat([latent_kv_cache, c_kv], dim=1)
        else:
            c_kv_full = c_kv

        L_full = c_kv_full.shape[1]

        # 4. Descompresión de K y V desde el espacio latente
        k_full = self.k_up_proj(c_kv_full).view(B, L_full, self.n_heads, self.head_dim).transpose(1, 2)
        v_full = self.v_up_proj(c_kv_full).view(B, L_full, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            k_full = self.k_norm(k_full)
        k_full = self.rope.apply_rope(k_full, start_pos=0)

        # 5. Atención punto-producto escalado
        if latent_kv_cache is not None:
            attn_out = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=False)
        else:
            eff_mask = attn_mask if attn_mask is not None else self._get_block_mask(L, x.device)
            attn_out = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=eff_mask)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.o_proj(attn_out, record_telemetry, integer_mode, add_sub_mode)

        # Retornar salida y la caché COMPRIMIDA c_kv_full (ahorro masivo de RAM)
        return out, c_kv_full
