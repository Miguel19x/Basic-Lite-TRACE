# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from trace.layers.ffn import TernaryExpert


class MemoryAwarePagedMoE(nn.Module):
    """
    MoE Residente de 4 Expertos + Experto Compartido (Basic-Lite TRACE):
    - 100% residente en RAM (sin sobrecoste de simulación de desalojo NVMe).
    - Router lineal con despacho de tokens por bloques causales.
    - Experto compartido permanente estilo DeepSeek-V3.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_experts = cfg.num_experts
        self.capacity_factor = cfg.capacity_factor
        self.block_size = cfg.block_size
        self.min_gate_threshold = 1e-4

        self.router = nn.Linear(cfg.d_model, cfg.num_experts, bias=False)

        # Repositorio de expertos ternarios residentes
        self.expert_bank = nn.ModuleList([
            TernaryExpert(cfg.d_model, cfg.d_ff_expert, eps=cfg.eps)
            for _ in range(cfg.num_experts)
        ])

        # Experto compartido permanente
        self.use_shared_expert = getattr(cfg, "use_shared_expert", True)
        if self.use_shared_expert:
            self.shared_expert = TernaryExpert(cfg.d_model, cfg.d_ff_expert, eps=cfg.eps)

        self.last_selected_experts: List[int] = []

    def forward(
        self,
        x: torch.Tensor,
        t_noise: torch.Tensor,
        non_pad_mask: Optional[torch.Tensor] = None,
        active_mask: Optional[torch.Tensor] = None,
        r_step: int = 0,
        record_telemetry: bool = False,
        integer_mode: bool = False,
        add_sub_mode: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        shared_out = 0.0
        if self.use_shared_expert:
            shared_out = self.shared_expert(x, record_telemetry, integer_mode, add_sub_mode)
            if non_pad_mask is not None:
                shared_out = shared_out * non_pad_mask.unsqueeze(-1).float()

        B_blk = min(self.block_size, L)
        num_blocks = L // B_blk
        num_total_blocks = B * num_blocks

        with torch.no_grad():
            mean_t = float(t_noise.detach().mean().item())
            dynamic_factor = self.capacity_factor * (1.25 - 0.50 * mean_t)
            k = max(1, min(B_blk, int(round((B_blk * dynamic_factor) / self.num_experts))))

        x_blk = x.reshape(num_total_blocks, B_blk, D)
        router_logits = self.router(x_blk)
        scores = F.softmax(router_logits, dim=-1)

        if active_mask is not None:
            act_blk = active_mask.reshape(num_total_blocks, B_blk, 1)
            scores = scores * (0.10 + 0.90 * act_blk)
            scores = scores * (act_blk > 0.05).float()

        if non_pad_mask is not None:
            valid_blk = non_pad_mask.reshape(num_total_blocks, B_blk, 1).float()
            scores = scores * valid_blk

        curr_node_ids = scores.argmax(dim=-1).detach()
        scores_t = scores.transpose(1, 2)
        top_gating, top_indices = torch.topk(scores_t, k=k, dim=-1)

        active_experts_in_step = set()
        for e_idx in range(self.num_experts):
            if (top_gating[:, e_idx, :] > self.min_gate_threshold).any():
                active_experts_in_step.add(e_idx)

        x_flat = x_blk.reshape(-1, D)
        out_flat = torch.zeros_like(x_flat)
        block_offsets = (torch.arange(num_total_blocks, device=x.device) * B_blk).unsqueeze(1)

        for e_idx in active_experts_in_step:
            exp_mod = self.expert_bank[e_idx]
            idx_e = top_indices[:, e_idx, :]
            gate_e = top_gating[:, e_idx, :]
            flat_idx = (block_offsets + idx_e).reshape(-1)
            flat_gate = gate_e.reshape(-1)

            valid_mask = flat_gate > self.min_gate_threshold
            active_idx = torch.nonzero(valid_mask, as_tuple=True)[0]
            num_active = int(active_idx.numel())

            if num_active > 0:
                token_indices = flat_idx.index_select(0, active_idx)
                gates_active = flat_gate.index_select(0, active_idx).unsqueeze(-1)
                selected = x_flat.index_select(0, token_indices)
                expert_out = exp_mod(selected, record_telemetry, integer_mode, add_sub_mode) * gates_active
                out_flat.index_add_(0, token_indices, expert_out)

        self.last_selected_experts = list(active_experts_in_step)
        return (out_flat.reshape(B, L, D) + shared_out), curr_node_ids

    def get_last_telemetry(self) -> Dict[str, Any]:
        sel = getattr(self, "last_selected_experts", [])
        return {
            "selected_experts": sel,
            "active_slot_ratio": len(sel) / max(1, self.num_experts),
        }
