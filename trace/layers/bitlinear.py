# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from trace.layers.norm import RMSNorm
from trace.ops.ternary_cpu import ternary_addsub_matmul


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
        quantize_weights_enabled: bool = True,
        weight_scale_type: str = "mean",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.use_sub_ln = use_sub_ln
        self.quantize_activations = quantize_activations
        self.quantize_weights_enabled = quantize_weights_enabled
        self.weight_scale_type = weight_scale_type
        self.sub_ln = RMSNorm(in_features, eps=eps) if use_sub_ln else None
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.register_buffer("int8_codes", torch.zeros(out_features, in_features, dtype=torch.int8), persistent=False)
        self.register_buffer("packed_scale", torch.tensor(1.0, dtype=torch.float32), persistent=False)
        self.register_buffer("cached_w_eff", torch.empty(0), persistent=False)
        self.register_buffer("prev_codes", torch.zeros(out_features, in_features, dtype=torch.int8), persistent=False)
        self.last_stats: dict[str, float] = {}
        self.use_loaded_bitcos: bool = False
        self._has_prev_codes: bool = False
        self.register_load_state_dict_post_hook(self._on_load_state_dict)

    @staticmethod
    def _on_load_state_dict(module: "BitLinear", _incompatible_keys: object) -> None:
        module.cached_w_eff = torch.empty(0, device=module.weight.device)
        module.use_loaded_bitcos = False

    def _scale_int32_output(
        self,
        acc_int32: torch.Tensor,
        scale_x: torch.Tensor,
        q_max: float,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        return (acc_int32.float() * ((scale_x.float() / q_max) * self.packed_scale.float())).to(out_dtype)

    def quantize_weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w_f32 = self.weight.float()
        if self.weight_scale_type == "median":
            scale_w = w_f32.abs().median().clamp(min=self.eps)
        else:
            scale_w = w_f32.abs().mean().clamp(min=self.eps)

        if not self.quantize_weights_enabled:
            empty_q = torch.empty(0, device=self.weight.device, dtype=torch.int8)
            return self.weight, empty_q, scale_w.detach()

        q_w = torch.clamp(torch.round(w_f32 / scale_w), -1.0, 1.0)
        w_eff = self.weight + ((scale_w * q_w).to(self.weight.dtype) - self.weight).detach()
        return w_eff, q_w.detach(), scale_w.detach()

    def get_effective_weight(self) -> torch.Tensor:
        if self.training and self.weight.numel() == 0:
            raise RuntimeError("No se puede entrenar una capa BitLinear cargada desde Paged BitCOS sin pesos FP32.")
        if self.use_loaded_bitcos:
            return self.int8_codes.to(self.packed_scale.dtype) * self.packed_scale
        if self.training:
            if self.cached_w_eff.numel() > 0:
                self.cached_w_eff = torch.empty(0, device=self.weight.device)
            return self.quantize_weights()[0]
        if not self.quantize_weights_enabled:
            return self.weight
        if self.cached_w_eff.numel() == 0:
            self.refresh_eval_cache()
        return self.cached_w_eff

    @torch.no_grad()
    def refresh_eval_cache(self) -> None:
        """Precalcula W_eff y códigos ternarios una sola vez al pasar a eval()."""
        if self.use_loaded_bitcos or not self.quantize_weights_enabled:
            return
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
        if self.training and self.weight.numel() == 0:
            raise RuntimeError("No se puede entrenar una capa BitLinear cargada desde Paged BitCOS sin pesos FP32.")

        x_norm = self.sub_ln(x) if (self.sub_ln is not None and self.use_sub_ln) else x

        # Fase continua de Continual QAT (16-bit -> 1.58-bit): evita cuantizar activaciones/pesos en warmup
        if not self.quantize_weights_enabled and not self.use_loaded_bitcos:
            return F.linear(x_norm, self.weight, bias=None)

        if not self.quantize_activations:
            w_eff = self.get_effective_weight()
            return F.linear(x_norm, w_eff.to(x_norm.dtype), bias=None)

        q_max = 127.0
        x_norm_f32 = x_norm.float()
        scale_x = x_norm_f32.abs().amax(dim=-1, keepdim=True).clamp(min=self.eps)

        # Blindaje para modelo cargado desde Paged BitCOS (0 pesos FP32 en RAM)
        if self.use_loaded_bitcos:
            q_x = torch.clamp(torch.round(x_norm_f32 * (q_max / scale_x)), -q_max, q_max)
            if add_sub_mode:
                acc_int32 = ternary_addsub_matmul(q_x.to(torch.int8), self.int8_codes)
                return self._scale_int32_output(acc_int32, scale_x, q_max, x.dtype)
            # Tanto para integer_mode como para evaluación por defecto
            q_x_int = q_x.to(torch.int32)
            w_codes_int = self.int8_codes.to(torch.int32)
            acc_int32 = torch.matmul(q_x_int, w_codes_int.t())
            return self._scale_int32_output(acc_int32, scale_x, q_max, x.dtype)

        if not self.training:
            q_x = torch.clamp(torch.round(x_norm_f32 * (q_max / scale_x)), -q_max, q_max)

            if add_sub_mode:
                if self.cached_w_eff.numel() == 0:
                    self.refresh_eval_cache()
                # Inferencia aditiva pura vía kernel nativo C OpenMP/AVX2
                acc_int32 = ternary_addsub_matmul(q_x.to(torch.int8), self.int8_codes)
                return self._scale_int32_output(acc_int32, scale_x, q_max, x.dtype)

            if integer_mode:
                if self.cached_w_eff.numel() == 0:
                    self.refresh_eval_cache()
                q_x_int = q_x.to(torch.int32)
                w_codes_int = self.int8_codes.to(torch.int32)
                acc_int32 = torch.matmul(q_x_int, w_codes_int.t())
                return self._scale_int32_output(acc_int32, scale_x, q_max, x.dtype)

            if self.cached_w_eff.numel() == 0:
                self.refresh_eval_cache()
            x_eff = (q_x * (scale_x / q_max)).to(x_norm.dtype)
            return F.linear(x_eff, self.cached_w_eff, bias=None)

        # Ruta de entrenamiento QAT con STE (invalida caché de evaluación si existía)
        if self.cached_w_eff.numel() > 0:
            self.cached_w_eff = torch.empty(0, device=x.device)

        q_x = torch.clamp(torch.round(x_norm_f32 * (q_max / scale_x)), -q_max, q_max)
        x_eff = x_norm + ((q_x * (scale_x / q_max)).to(x_norm.dtype) - x_norm).detach()
        w_eff, q_w, scale_w = self.quantize_weights()

        if record_telemetry:
            with torch.no_grad():
                codes_i8 = q_w.to(torch.int8)
                total = float(q_w.numel())
                zeros = float((codes_i8 == 0).sum().item()) / total
                pos = float((codes_i8 == 1).sum().item()) / total
                neg = float((codes_i8 == -1).sum().item()) / total
                if self._has_prev_codes:
                    flips = float((codes_i8 != self.prev_codes).sum().item()) / total
                else:
                    flips = 0.0
                    self._has_prev_codes = True
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
