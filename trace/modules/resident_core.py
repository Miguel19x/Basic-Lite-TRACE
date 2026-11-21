# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from trace.core.weight_manager import WeightMovementManager
from trace.layers.bitlinear import BitLinear
from trace.layers.embeddings import SinusoidalStepEncoding
from trace.layers.norm import RMSNorm
from trace.modules.attention import BlockCausalGQATernaryAttention
from trace.modules.mla_attention import MultiHeadLatentAttention
from trace.modules.paged_moe import MemoryAwarePagedMoE


class SharedResidentTernaryCore(nn.Module):
    """
    Núcleo ternario compartido residente de Basic-Lite TRACE:
    - Multi-Head Latent Attention (MLA) o Block-Causal GQA según sistema de activación.
    - MoE con paginación de expertos y enrutamiento consciente de memoria RAM/SSD.
    - Dinámica dual de razonamiento (z_r) y respuesta (y_r) con escala residual compartida.
    """

    def __init__(self, cfg, weight_manager: Optional[WeightMovementManager] = None):
        super().__init__()
        self.cfg = cfg
        self.use_mla = getattr(cfg, "use_mla", False)
        self.norm_z = RMSNorm(cfg.d_model, eps=cfg.eps)
        self.norm_y = RMSNorm(cfg.d_model, eps=cfg.eps)
        self.norm_moe = RMSNorm(cfg.d_model, eps=cfg.eps)
        self.norm_refine = RMSNorm(cfg.d_model, eps=cfg.eps)

        self.anchor_proj = BitLinear(cfg.d_model, cfg.d_model, eps=cfg.eps)
        if self.use_mla:
            self.attn = MultiHeadLatentAttention(cfg)
        else:
            self.attn = BlockCausalGQATernaryAttention(cfg)
        self.paged_moe = MemoryAwarePagedMoE(cfg, weight_manager=weight_manager)
        self.answer_updater = BitLinear(cfg.d_model, cfg.d_model, eps=cfg.eps)

        self.halt_gate = nn.Linear(cfg.d_model, 1)
        nn.init.constant_(self.halt_gate.bias, -1.5)

        self.step_encoder = SinusoidalStepEncoding(cfg.d_model)
        self.noise_mlp = nn.Sequential(
            nn.Linear(2, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.alpha_z = nn.Parameter(torch.tensor(0.60))
        self.beta_y = nn.Parameter(torch.tensor(0.60))
        self.carry_gate = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        z_r: torch.Tensor,
        y_r: torch.Tensor,
        anchor_feat: torch.Tensor,
        t_noise: torch.Tensor,
        delta_t: torch.Tensor,
        r_step: int,
        total_recursions: int,
        non_pad_mask: Optional[torch.Tensor],
        active_mask: torch.Tensor,
        latent_kv_cache: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        record_telemetry: bool = False,
        integer_mode: bool = False,
        add_sub_mode: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if t_noise.dim() == 1:
            noise_pair = torch.stack([t_noise, delta_t], dim=-1)
            e_t = self.noise_mlp(noise_pair).unsqueeze(1)
        elif t_noise.dim() == 2:
            L_in = t_noise.shape[1]
            L_core = z_r.shape[1]
            if L_in != L_core and L_in % L_core == 0:
                step_down = L_in // L_core
                t_down = t_noise[:, ::step_down]
                dt_down = delta_t[:, ::step_down] if delta_t.dim() == 2 else delta_t
            else:
                t_down, dt_down = t_noise, delta_t
            noise_pair = torch.stack([t_down, dt_down], dim=-1)
            e_t = self.noise_mlp(noise_pair)
        else:
            e_t = self.noise_mlp(torch.stack([t_noise, delta_t], dim=-1))

        u_norm = float(r_step + 1) / float(max(1, total_recursions))
        step_scale = 2.0 / float(total_recursions + 1)
        e_r = self.step_encoder(u_norm, z_r.device)

        joint_state = self.norm_z(z_r) + self.norm_y(y_r) + anchor_feat + e_t + e_r

        # Atención: Multi-Head Latent Attention (MLA) o BlockCausalGQATernaryAttention según activación
        if self.use_mla:
            attn_out, updated_latent_kv = self.attn(
                joint_state,
                latent_kv_cache=latent_kv_cache,
                start_pos=start_pos,
                record_telemetry=record_telemetry,
                integer_mode=integer_mode,
                add_sub_mode=add_sub_mode,
                attn_mask=attn_mask,
            )
        else:
            attn_out, updated_latent_kv = self.attn(
                joint_state,
                kv_cache=latent_kv_cache,
                start_pos=start_pos,
                record_telemetry=record_telemetry,
                integer_mode=integer_mode,
                add_sub_mode=add_sub_mode,
                attn_mask=attn_mask,
            )
        eff_alpha = self.alpha_z * step_scale
        eff_beta = self.beta_y * step_scale

        z_mid = z_r + active_mask * eff_alpha * attn_out

        # Memory-Aware Paged MoE
        moe_out, curr_node_ids = self.paged_moe(
            self.norm_moe(z_mid),
            t_noise=t_noise,
            non_pad_mask=non_pad_mask,
            active_mask=active_mask,
            r_step=r_step,
            record_telemetry=record_telemetry,
            integer_mode=integer_mode,
            add_sub_mode=add_sub_mode,
        )
        z_next = z_mid + active_mask * eff_alpha * moe_out

        fusion_yz = self.norm_refine(y_r + z_next)
        delta_y = self.answer_updater(fusion_yz, record_telemetry, integer_mode, add_sub_mode)
        y_next = y_r + active_mask * eff_beta * F.silu(delta_y)

        halt_prob = torch.sigmoid(self.halt_gate(self.norm_refine(z_next + y_next)))
        return z_next, y_next, halt_prob, updated_latent_kv, curr_node_ids
