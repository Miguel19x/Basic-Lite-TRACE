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
from trace.modules.moe import TernaryExpert


class MemoryAwarePagedMoE(nn.Module):
    """
    MoE con Enrutamiento Consciente de Memoria y Paginación de Expertos:
    - Aplica la penalización de transferencia S(e) = U(e) - lambda * C_lectura(e) - mu * C_computo(e).
    - Mecanismo de escape: Si un experto no residente tiene utilidad suficiente, se carga de almacenamiento.
    - Se ejecuta bajo un presupuesto estricto de RAM (solo N expertos residentes simultáneos en RAM).
    - Soporte para prefetching del siguiente paso.
    """

    def __init__(self, cfg, weight_manager: Optional[WeightMovementManager] = None):
        super().__init__()
        self.cfg = cfg
        self.num_experts = cfg.num_experts
        self.capacity_factor = cfg.capacity_factor
        patch_sz = getattr(cfg, "byte_patch_size", 1) if getattr(cfg, "use_byte_patching", True) else 1
        self.block_size = max(1, cfg.block_size // patch_sz)
        self.min_gate_threshold = getattr(cfg, "min_gate_threshold", 1e-4)

        self.router = nn.Linear(cfg.d_model, cfg.num_experts, bias=False)

        # Repositorio de expertos ternarios
        self.expert_bank = nn.ModuleList([
            TernaryExpert(cfg.d_model, cfg.d_ff_expert, eps=cfg.eps)
            for _ in range(cfg.num_experts)
        ])

        # Experto compartido permanente (100% residente en RAM estilo DeepSeek-V3)
        self.use_shared_expert = getattr(cfg, "use_shared_expert", True)
        if self.use_shared_expert:
            self.shared_expert = TernaryExpert(cfg.d_model, cfg.d_ff_expert, eps=cfg.eps)

        # Gestor de pesos y caché de memoria
        self.enable_nvme_io_simulation = getattr(cfg, "enable_nvme_io_simulation", False)
        if weight_manager is None:
            max_res = getattr(cfg, "max_resident_experts", 2) if self.enable_nvme_io_simulation else cfg.num_experts
            l_cost = getattr(cfg, "lambda_read_cost", 0.35)
            m_cost = getattr(cfg, "mu_compute_cost", 0.05)
            esc_thr = getattr(cfg, "memory_escape_threshold", 0.70)
            self.weight_manager = WeightMovementManager(
                max_resident_experts=max_res,
                lambda_read_cost=l_cost,
                mu_compute_cost=m_cost,
                escape_threshold=esc_thr,
            )
        else:
            self.weight_manager = weight_manager

        # Registrar expertos en el gestor de pesos
        for e_idx, exp in enumerate(self.expert_bank):
            self.weight_manager.register_expert_source(e_idx, exp)

        # Pre-cargar los primeros N expertos en RAM
        for e_idx in range(min(self.num_experts, self.weight_manager.max_resident_experts)):
            self.weight_manager.acquire_expert(e_idx)

        self.last_selected_experts: List[int] = []

    def forward(
        self,
        x: torch.Tensor,
        t_noise: torch.Tensor,
        non_pad_mask: Optional[torch.Tensor] = None,
        active_mask: Optional[torch.Tensor] = None,
        r_step: int = 0,
        enable_prefetch: bool = True,
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
        raw_scores = F.softmax(router_logits, dim=-1)  # [num_total_blocks, B_blk, E]

        # Enrutamiento de expertos:
        if self.enable_nvme_io_simulation:
            # Modo 30B: Aplicar Enrutamiento Consciente de Memoria S(e) y simulación E/S
            block_utilities = raw_scores.mean(dim=1)  # [num_total_blocks, E]
            adjusted_utilities, escaped = self.weight_manager.compute_memory_aware_scores(
                block_utilities,
                allow_escape=True,
            )
            scores_weight = (adjusted_utilities / block_utilities.clamp(min=1e-5)).unsqueeze(1)
            scores = (raw_scores * scores_weight).clamp(min=0.0)
            scores = scores / scores.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        else:
            # Modo Directo RAM (Núcleo 870K): Todos los expertos residentes, sin sobrecoste de simulación E/S
            scores = raw_scores

        if active_mask is not None:
            act_blk = active_mask.reshape(num_total_blocks, B_blk, 1)
            scores = scores * (0.10 + 0.90 * act_blk)
            scores = scores * (act_blk > 0.05).float()

        if non_pad_mask is not None:
            valid_blk = non_pad_mask.reshape(num_total_blocks, B_blk, 1).float()
            scores = scores * valid_blk

        curr_node_ids = scores.argmax(dim=-1).detach()
        scores_t = scores.transpose(1, 2)  # [num_total_blocks, E, B_blk]
        top_gating, top_indices = torch.topk(scores_t, k=k, dim=-1)

        # Identificar qué expertos fueron seleccionados con gating significativo
        active_experts_in_step = set()
        for e_idx in range(self.num_experts):
            if (top_gating[:, e_idx, :] > self.min_gate_threshold).any():
                active_experts_in_step.add(e_idx)

        # Cargar los módulos expertos a ejecutar
        executed_modules = {}
        if self.enable_nvme_io_simulation:
            for e_idx in active_experts_in_step:
                exp_mod = self.weight_manager.acquire_expert(e_idx)
                executed_modules[e_idx] = exp_mod
        else:
            for e_idx in active_experts_in_step:
                executed_modules[e_idx] = self.expert_bank[e_idx]

        # Anticipación (Prefetch): Identificar el segundo experto más probable y solicitar prefetch en modo 30B
        if self.enable_nvme_io_simulation and enable_prefetch and len(active_experts_in_step) > 0:
            avg_scores = scores.mean(dim=(0, 1))
            sorted_e = torch.argsort(avg_scores, descending=True)
            for cand in sorted_e.tolist():
                if cand not in active_experts_in_step:
                    self.weight_manager.prefetch_expert(cand)
                    break

        # Ejecución sobre los tokens seleccionados
        x_flat = x_blk.reshape(-1, D)
        out_flat = torch.zeros_like(x_flat)
        block_offsets = (torch.arange(num_total_blocks, device=x.device) * B_blk).unsqueeze(1)

        for e_idx, exp_mod in executed_modules.items():
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
