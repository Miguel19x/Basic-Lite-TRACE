# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

import math
from trace.layers.norm import RMSNorm
from trace.ops.ternary_cpu import ternary_addsub_matmul

import torch
import torch.nn as nn
import torch.nn.functional as F


class BitLinear(nn.Module):
    """
    Capa BitLinear W1.58A8 con:
    - Entrenamiento QAT mediante Straight-Through Estimator (STE).
    - Caché W_eff en eval() para benchmarks directos en FP32.
    - Modo de inferencia aditiva/sustractiva pura acelerada por kernel nativo C (AVX2/OpenMP).
    - Modo entero INT8 con acumulación INT32.
    - Serialización y deserialización BITCOSv2 de 2 bits empaquetados por peso.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        eps: float = 1e-6,
        use_sub_ln: bool = True,
        quantize_activations: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.use_sub_ln = use_sub_ln
        self.quantize_activations = quantize_activations
        self.sub_ln = RMSNorm(in_features, eps=eps) if use_sub_ln else None
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.register_buffer("int8_codes", torch.zeros(out_features, in_features, dtype=torch.int8), persistent=False)
        self.register_buffer("packed_scale", torch.tensor(1.0, dtype=torch.float32), persistent=False)
        self.register_buffer("cached_w_eff", torch.empty(0), persistent=False)
        self.register_buffer("prev_codes", torch.zeros(out_features, in_features, dtype=torch.int8), persistent=False)
        self.last_stats: dict[str, float] = {}
        self.use_loaded_bitcos: bool = False

    def quantize_weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_w = self.weight.abs().mean().clamp(min=self.eps)
        q_w = torch.clamp(torch.round(self.weight / scale_w), -1.0, 1.0)
        w_eff = self.weight + (scale_w * q_w - self.weight).detach()
        return w_eff, q_w.detach(), scale_w.detach()

    def get_effective_weight(self) -> torch.Tensor:
        """Devuelve el tensor de pesos efectivo W_eff (preservando STE en train o caché en eval)."""
        if self.training:
            if self.cached_w_eff.numel() > 0:
                self.cached_w_eff = torch.empty(0, device=self.weight.device)
            return self.quantize_weights()[0]

        if self.use_loaded_bitcos:
            return self.int8_codes.to(self.packed_scale.dtype) * self.packed_scale

        if self.cached_w_eff.numel() == 0:
            self.refresh_eval_cache()
        return self.cached_w_eff

    @torch.no_grad()
    def refresh_eval_cache(self) -> None:
        """Precalcula W_eff y códigos ternarios una sola vez al pasar a eval()."""
        w_eff, q_w, scale_w = self.quantize_weights()
        self.cached_w_eff = w_eff.detach().clone()
        self.int8_codes.copy_(q_w.to(torch.int8))
        self.packed_scale.fill_(float(scale_w.item()))

    def forward(
        self,
        x: torch.Tensor,
        record_telemetry: bool = False,
        integer_mode: bool = False,
        add_sub_mode: bool = False,
    ) -> torch.Tensor:
        x_norm = self.sub_ln(x) if (self.sub_ln is not None and self.use_sub_ln) else x

        if not self.quantize_activations:
            w_eff = self.get_effective_weight()
            return F.linear(x_norm, w_eff, bias=None)
        q_max = 127.0
        scale_x = x_norm.abs().amax(dim=-1, keepdim=True).clamp(min=self.eps)

        if not self.training:
            q_x = torch.clamp(torch.round(x_norm * (q_max / scale_x)), -q_max, q_max)

            # Blindaje para modelo cargado desde Paged BitCOS (0 pesos FP32 en RAM)
            if self.use_loaded_bitcos:
                if add_sub_mode:
                    acc_int32 = ternary_addsub_matmul(q_x.to(torch.int8), self.int8_codes)
                    return acc_int32.to(x.dtype) * ((scale_x / q_max) * self.packed_scale)
                # Tanto para integer_mode como para evaluación por defecto
                q_x_int = q_x.to(torch.int32)
                w_codes_int = self.int8_codes.to(torch.int32)
                acc_int32 = torch.matmul(q_x_int, w_codes_int.t())
                return acc_int32.to(x.dtype) * ((scale_x / q_max) * self.packed_scale)

            if add_sub_mode:
                if self.cached_w_eff.numel() == 0:
                    self.refresh_eval_cache()
                # Inferencia aditiva pura vía kernel nativo C OpenMP/AVX2
                acc_int32 = ternary_addsub_matmul(q_x.to(torch.int8), self.int8_codes)
                return acc_int32.to(x.dtype) * ((scale_x / q_max) * self.packed_scale)

            if integer_mode:
                if self.cached_w_eff.numel() == 0:
                    self.refresh_eval_cache()
                q_x_int = q_x.to(torch.int32)
                w_codes_int = self.int8_codes.to(torch.int32)
                acc_int32 = torch.matmul(q_x_int, w_codes_int.t())
                return acc_int32.to(x.dtype) * ((scale_x / q_max) * self.packed_scale)

            if self.cached_w_eff.numel() == 0:
                self.refresh_eval_cache()
            x_eff = q_x * (scale_x / q_max)
            return F.linear(x_eff, self.cached_w_eff, bias=None)

        # Ruta de entrenamiento QAT con STE (invalida caché de evaluación si existía)
        if self.cached_w_eff.numel() > 0:
            self.cached_w_eff = torch.empty(0, device=x.device)

        q_x = torch.clamp(torch.round(x_norm * (q_max / scale_x)), -q_max, q_max)
        x_eff = x_norm + (q_x * (scale_x / q_max) - x_norm).detach()
        w_eff, q_w, scale_w = self.quantize_weights()

        if record_telemetry:
            with torch.no_grad():
                codes_i8 = q_w.to(torch.int8)
                total = float(q_w.numel())
                zeros = float((codes_i8 == 0).sum().item()) / total
                pos = float((codes_i8 == 1).sum().item()) / total
                neg = float((codes_i8 == -1).sum().item()) / total
                flips = float((codes_i8 != self.prev_codes).sum().item()) / total
                sat_ratio = float((q_x.abs() == q_max).float().mean().item())

                self.prev_codes.copy_(codes_i8)
                self.last_stats = {
                    "zero_frac": zeros,
                    "pos_frac": pos,
                    "neg_frac": neg,
                    "bitcos_bpw": 2.0 - zeros,
                    "scale_w": float(scale_w.item()),
                    "act_saturation": sat_ratio,
                    "code_flip_rate": flips,
                }

        return F.linear(x_eff, w_eff, bias=None)
