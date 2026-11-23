# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

import random
from trace.layers.abacus import AbacusEmbedding
from trace.layers.bitlinear import BitLinear
from trace.layers.norm import RMSNorm
from trace.modules.attention import build_target_unified_mask
from trace.modules.resident_core import SharedResidentTernaryCore
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class TRACECausalModel(nn.Module):
    """
    TRACE Causal: Arquitectura Causal de 2 Etapas con Abacus Embeddings (Basic-Lite TRACE).
    
    Innovaciones Arquitectónicas:
    1. Abacus Positional Embeddings:
       Codifica el rango decimal de cada dígito (unidades=0, decenas=1, centenas=2)
       otorgando conciencia de valor posicional a los cabezales de atención.
    2. Factorización Causal en 2 Pasadas (Two-Stage Causal Field):
       - Etapa 1: Predicción de subtotales (multiplicación base Pw, Pms).
       - Anclaje Causal: Los subtotales se inyectan como hechos consumados del contexto.
       - Etapa 2: Predicción de totales finales (adición multivariable watt, ms).
    3. Mitigación de Sesgo de Exposición:
       Entrenamiento con Scheduled Sampling para robustez ante errores de la Etapa 1.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # 1. Capas de Entrada
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.abacus_emb = AbacusEmbedding(
            cfg.d_model,
            max_ranks=16,
            delimiter_rank=15,
            mask_token_id=cfg.mask_token_id,
        )
        self.stage_emb = nn.Embedding(4, cfg.d_model)

        # 2. Núcleo Ternario Recursivo Residente (Backbone con BitLinear W1.58A8)
        self.resident_core = SharedResidentTernaryCore(cfg)

        # 3. Normalización y Cabezal de Lenguaje (FP32 para rango dinámico completo en 10 dígitos)
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def configure_adamw_groups(self, bitlinear_weight_decay: float = 0.0) -> list[dict]:
        """
        Configuración de decaimiento desacoplado en 3 grupos (BitNet b1.58):
        1. dense_decay: matrices densas FP32 (ej. lm_head) con weight_decay estándar.
        2. bitlinear_shadow: shadow weights de BitLinear con decaimiento dinámico de etapa.
        3. no_decay: normas, embeddings, compuertas y parámetros 1D con 0.0.
        """
        bitlinear_param_ids = {
            id(p) for m in self.modules() if isinstance(m, BitLinear) for p in m.parameters()
        }
        decay, bitlinear_shadow, no_decay = [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in bitlinear_param_ids:
                if param.ndim >= 2:
                    bitlinear_shadow.append(param)
                else:
                    no_decay.append(param)
            elif (
                param.ndim < 2
                or "emb" in name
                or "norm" in name
                or "alpha" in name
                or "beta" in name
                or "gate" in name
            ):
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {"params": decay, "weight_decay": self.cfg.weight_decay, "name": "dense_decay"},
            {"params": bitlinear_shadow, "weight_decay": bitlinear_weight_decay, "name": "bitlinear_shadow"},
            {"params": no_decay, "weight_decay": 0.0, "name": "no_decay"},
        ]

    def compute_input_embeddings(self, x_t: torch.Tensor) -> torch.Tensor:
        """Suma de incrustación de token más incrustación de ábaco decimal."""
        tok_feat = self.token_emb(x_t)
        abacus_feat = self.abacus_emb(x_t)
        return tok_feat + abacus_feat

    def forward_backbone(
        self,
        x_t: torch.Tensor,
        t_noise: torch.Tensor,
        num_recursions: int | None = None,
        stage_id: int = 0,
        attn_mask: torch.Tensor | None = None,
        eval_mask: torch.Tensor | None = None,
        record_telemetry: bool = False,
    ) -> torch.Tensor:
        """
        Ejecuta el paso hacia adelante a través del núcleo recursivo ternario.
        Soporta recorte dinámico de padding, condicionamiento de etapa (stage_id)
        y proyección perezosa de vocabulario sobre eval_mask.
        """
        B, L = x_t.shape
        non_pad_mask = (x_t != self.cfg.pad_token_id)

        # Recorte dinámico de padding al múltiplo de block_size más cercano
        max_valid = int(non_pad_mask.sum(dim=1).max().item())
        bs = self.cfg.block_size
        L_trim = min(L, max(bs, ((max_valid + bs - 1) // bs) * bs))

        if L_trim < L:
            x_t = x_t[:, :L_trim].contiguous()
            non_pad_mask = non_pad_mask[:, :L_trim].contiguous()
            if t_noise.dim() == 2 and t_noise.shape[1] == L:
                t_noise = t_noise[:, :L_trim].contiguous()
            if attn_mask is not None and attn_mask.shape[-1] == L:
                attn_mask = attn_mask[..., :L_trim, :L_trim].contiguous()
            if eval_mask is not None and eval_mask.shape[-1] == L:
                eval_mask = eval_mask[:, :L_trim].contiguous()
            L = L_trim

        R = num_recursions if num_recursions is not None else self.cfg.num_recursions
        delta_t = torch.full_like(t_noise, getattr(self.cfg, "delta_t_val", 0.25))

        # Incrustación combinada con Abacus y Condicionamiento de Etapa
        a = self.compute_input_embeddings(x_t)
        a = a + self.stage_emb.weight[stage_id]

        z_r = a.clone()
        y_r = torch.zeros_like(a)

        int_mode = getattr(self.cfg, "use_integer_inference", False)
        add_sub = getattr(self.cfg, "use_add_sub_inference", False)

        anchor_feat = self.resident_core.anchor_proj(
            a, record_telemetry=record_telemetry, integer_mode=int_mode, add_sub_mode=add_sub
        )

        active_mask = torch.ones(B, L, 1, device=x_t.device)

        for r in range(R):
            z_next, y_next, _, _, _ = self.resident_core(
                z_r,
                y_r,
                anchor_feat,
                t_noise,
                delta_t,
                r,
                R,
                non_pad_mask,
                active_mask,
                latent_kv_cache=None,
                start_pos=0,
                record_telemetry=(record_telemetry and r == 0),
                integer_mode=int_mode,
                add_sub_mode=add_sub,
                attn_mask=attn_mask,
            )
            z_r, y_r = z_next, y_next

        normed_y = self.final_norm(y_r)
        if self.cfg.lazy_vocab_projection and eval_mask is not None:
            return self.lm_head(normed_y[eval_mask])
        return self.lm_head(normed_y)

    def forward_two_stage(
        self,
        x_0: torch.Tensor,
        prompt_lens: list[int],
        t_noise: torch.Tensor,
        scheduled_sampling_p: float = 0.85,
        record_telemetry: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Paso hacia adelante de 2 etapas para entrenamiento conjunto.
        
        Etapa 1: Enmascara subtotales y totales. Predice subtotales (pw, pms).
        Etapa 2: Ancla subtotales predichos (o ground truth con prob p). Predice totales (watt, ms).
        """
        B, L = x_0.shape
        device = x_0.device

        # Construir máscara de Etapa 1: Subtotales y Totales enmascarados
        x_stage1 = x_0.clone()
        stage1_digit_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        stage2_digit_mask = torch.zeros(B, L, dtype=torch.bool, device=device)

        pw_spans = []
        pms_spans = []
        watt_spans = []
        ms_spans = []

        for b in range(B):
            plen = prompt_lens[b]
            # Offsets exactos del dataset con scratchpad
            pw_range = (plen + 4, plen + 7)
            pms_range = (plen + 13, plen + 16)
            watt_range = (plen + 23, plen + 26)
            ms_range = (plen + 31, plen + 34)

            pw_spans.append(pw_range)
            pms_spans.append(pms_range)
            watt_spans.append(watt_range)
            ms_spans.append(ms_range)

            # En Etapa 1, todos los dígitos de target están enmascarados
            x_stage1[b, pw_range[0]:pw_range[1]] = self.cfg.mask_token_id
            x_stage1[b, pms_range[0]:pms_range[1]] = self.cfg.mask_token_id
            x_stage1[b, watt_range[0]:watt_range[1]] = self.cfg.mask_token_id
            x_stage1[b, ms_range[0]:ms_range[1]] = self.cfg.mask_token_id

            # Máscaras de supervisión
            stage1_digit_mask[b, pw_range[0]:pw_range[1]] = True
            stage1_digit_mask[b, pms_range[0]:pms_range[1]] = True

            stage2_digit_mask[b, watt_range[0]:watt_range[1]] = True
            stage2_digit_mask[b, ms_range[0]:ms_range[1]] = True

        # Construir máscara con bloque de target unificado bidireccional
        attn_mask = build_target_unified_mask(prompt_lens, L, self.cfg.block_size, device)

        # --- EJECUCIÓN ETAPA 1 (Multiplicaciones Subtotales) ---
        logits_stage1 = self.forward_backbone(
            x_stage1, t_noise, stage_id=0, attn_mask=attn_mask, eval_mask=stage1_digit_mask,
            record_telemetry=record_telemetry
        )

        # Pérdida de Etapa 1
        if logits_stage1.dim() == 3:
            L_curr = logits_stage1.shape[1]
            loss_stage1 = F.cross_entropy(logits_stage1[stage1_digit_mask[:, :L_curr]], x_0[stage1_digit_mask])
        else:
            loss_stage1 = F.cross_entropy(logits_stage1, x_0[stage1_digit_mask])

        # --- ANCLAJE CAUSAL PARA ETAPA 2 ---
        x_stage2 = x_stage1.clone()
        preds_stage1 = torch.zeros(B, L, dtype=torch.long, device=device)
        if logits_stage1.dim() == 3:
            preds_stage1[:, :logits_stage1.shape[1]] = logits_stage1.argmax(dim=-1).detach()
        else:
            preds_stage1[stage1_digit_mask] = logits_stage1.argmax(dim=-1).detach()

        for b in range(B):
            pw_r = pw_spans[b]
            pms_r = pms_spans[b]

            # Scheduled Sampling: Usar Ground Truth con prob p, o predicción propia con prob (1-p)
            use_gt = (random.random() < scheduled_sampling_p) if self.training else False
            if use_gt:
                x_stage2[b, pw_r[0]:pw_r[1]] = x_0[b, pw_r[0]:pw_r[1]]
                x_stage2[b, pms_r[0]:pms_r[1]] = x_0[b, pms_r[0]:pms_r[1]]
            else:
                x_stage2[b, pw_r[0]:pw_r[1]] = preds_stage1[b, pw_r[0]:pw_r[1]]
                x_stage2[b, pms_r[0]:pms_r[1]] = preds_stage1[b, pms_r[0]:pms_r[1]]

        # Desacoplamiento de gradientes para evitar retropropagación destructiva a través del anclaje
        x_stage2 = x_stage2.detach()

        # --- ETAPA 2: CAMPO CAUSAL POR COLUMNAS (UNIDADES -> DECENAS -> CENTENAS) ---
        # Sub-pasada 2.1: Predicción de Unidades (offset +2 de cada rango)
        mask_units = torch.zeros(B, L, dtype=torch.bool, device=device)
        for b in range(B):
            mask_units[b, watt_spans[b][0] + 2] = True
            mask_units[b, ms_spans[b][0] + 2] = True
        logits_units = self.forward_backbone(
            x_stage2, t_noise, stage_id=1, attn_mask=attn_mask, eval_mask=mask_units
        )
        if logits_units.dim() == 3:
            L_curr = logits_units.shape[1]
            loss_units = F.cross_entropy(logits_units[mask_units[:, :L_curr]], x_0[mask_units])
        else:
            loss_units = F.cross_entropy(logits_units, x_0[mask_units])

        # Inyección causal de Unidades
        preds_units = torch.zeros(B, L, dtype=torch.long, device=device)
        if logits_units.dim() == 3:
            preds_units[:, :logits_units.shape[1]] = logits_units.argmax(dim=-1).detach()
        else:
            preds_units[mask_units] = logits_units.argmax(dim=-1).detach()

        x_tens = x_stage2.clone()
        for b in range(B):
            u_gt = (random.random() < scheduled_sampling_p) if self.training else False
            for r_span in (watt_spans[b], ms_spans[b]):
                u_idx = r_span[0] + 2
                x_tens[b, u_idx] = x_0[b, u_idx] if u_gt else preds_units[b, u_idx]
        x_tens = x_tens.detach()

        # Sub-pasada 2.2: Predicción de Decenas (offset +1 de cada rango)
        mask_tens = torch.zeros(B, L, dtype=torch.bool, device=device)
        for b in range(B):
            mask_tens[b, watt_spans[b][0] + 1] = True
            mask_tens[b, ms_spans[b][0] + 1] = True
        logits_tens = self.forward_backbone(
            x_tens, t_noise, stage_id=2, attn_mask=attn_mask, eval_mask=mask_tens
        )
        if logits_tens.dim() == 3:
            L_curr = logits_tens.shape[1]
            loss_tens = F.cross_entropy(logits_tens[mask_tens[:, :L_curr]], x_0[mask_tens])
        else:
            loss_tens = F.cross_entropy(logits_tens, x_0[mask_tens])

        # Inyección causal de Decenas
        preds_tens = torch.zeros(B, L, dtype=torch.long, device=device)
        if logits_tens.dim() == 3:
            preds_tens[:, :logits_tens.shape[1]] = logits_tens.argmax(dim=-1).detach()
        else:
            preds_tens[mask_tens] = logits_tens.argmax(dim=-1).detach()

        x_hundreds = x_tens.clone()
        for b in range(B):
            t_gt = (random.random() < scheduled_sampling_p) if self.training else False
            for r_span in (watt_spans[b], ms_spans[b]):
                t_idx = r_span[0] + 1
                x_hundreds[b, t_idx] = x_0[b, t_idx] if t_gt else preds_tens[b, t_idx]
        x_hundreds = x_hundreds.detach()

        # Sub-pasada 2.3: Predicción de Centenas (offset +0 de cada rango)
        mask_hundreds = torch.zeros(B, L, dtype=torch.bool, device=device)
        for b in range(B):
            mask_hundreds[b, watt_spans[b][0]] = True
            mask_hundreds[b, ms_spans[b][0]] = True
        logits_hundreds = self.forward_backbone(
            x_hundreds, t_noise, stage_id=3, attn_mask=attn_mask, eval_mask=mask_hundreds
        )
        if logits_hundreds.dim() == 3:
            L_curr = logits_hundreds.shape[1]
            loss_hundreds = F.cross_entropy(logits_hundreds[mask_hundreds[:, :L_curr]], x_0[mask_hundreds])
        else:
            loss_hundreds = F.cross_entropy(logits_hundreds, x_0[mask_hundreds])

        # Rebalanceo simétrico 1:1 entre Etapa 1 y Etapa 2
        loss_stage2 = (loss_units + loss_tens + loss_hundreds) / 3.0
        total_loss = loss_stage1 + loss_stage2

        return {
            "loss": total_loss,
            "loss_stage1": loss_stage1,
            "loss_stage2": loss_stage2,
            "loss_units": loss_units,
            "loss_tens": loss_tens,
            "loss_hundreds": loss_hundreds,
            "logits_stage1": logits_stage1,
            "logits_units": logits_units,
            "logits_tens": logits_tens,
            "logits_hundreds": logits_hundreds,
            "stage1_mask": stage1_digit_mask,
            "stage2_mask": stage2_digit_mask,
        }

    @torch.no_grad()
    def generate_causal_two_step(
        self,
        prompt_bytes: bytes,
        override_scratchpad_tokens: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Inferencia Causal Determinista por Columnas Decimales (Zero Exposure Bias):
        1. Emite subtotales Pw, Pms (o utiliza override_scratchpad_tokens para auditoría causal).
        2. Fija Pw, Pms en la secuencia de entrada como hechos consumados del contexto.
        3. Campo Causal Decimal de Derecha a Izquierda:
           3.1. Emite unidades de watt y ms.
           3.2. Fija unidades en contexto y emite decenas (acarreo_0 resuelto).
           3.3. Fija decenas en contexto y emite centenas (acarreo_1 resuelto).
        """
        self.eval()
        device = next(self.parameters()).device

        # Construir plantilla de ancho fijo con scratchpad
        # pw: ???\npms: ???\nwatt: ???\nms: ???\n
        template_target = b"pw: ???\npms: ???\nwatt: ???\nms: ???\n"
        full_bytes = [self.cfg.bos_token_id] + list(prompt_bytes) + list(template_target) + [self.cfg.eos_token_id]
        prompt_len = 1 + len(prompt_bytes)

        # Padding a max_seq_len
        pad_len = max(0, self.cfg.max_seq_len - len(full_bytes))
        full_seq = full_bytes[:self.cfg.max_seq_len] + [self.cfg.pad_token_id] * pad_len
        x = torch.tensor([full_seq], dtype=torch.long, device=device)

        # Posiciones de los campos
        pw_start, pw_end = prompt_len + 4, prompt_len + 7
        pms_start, pms_end = prompt_len + 13, prompt_len + 16
        watt_start, watt_end = prompt_len + 23, prompt_len + 26
        ms_start, ms_end = prompt_len + 31, prompt_len + 34

        # Enmascarar todos los dígitos en Etapa 1
        x[0, pw_start:pw_end] = self.cfg.mask_token_id
        x[0, pms_start:pms_end] = self.cfg.mask_token_id
        x[0, watt_start:watt_end] = self.cfg.mask_token_id
        x[0, ms_start:ms_end] = self.cfg.mask_token_id

        t_noise = torch.ones(1, device=device)
        attn_mask = build_target_unified_mask([prompt_len], x.shape[1], self.cfg.block_size, device)

        # --- PASADA 1: Predecir Subtotales ---
        logits_1 = self.forward_backbone(x, t_noise, stage_id=0, attn_mask=attn_mask)
        probs_pw = F.softmax(logits_1[0, pw_start:pw_end], dim=-1)
        probs_pms = F.softmax(logits_1[0, pms_start:pms_end], dim=-1)
        pred_pw_tokens = logits_1[0, pw_start:pw_end].argmax(dim=-1)
        pred_pms_tokens = logits_1[0, pms_start:pms_end].argmax(dim=-1)

        # --- ANCLAJE CAUSAL (Inyección en contexto para Etapa 2) ---
        if override_scratchpad_tokens is not None:
            actual_pw_tokens, actual_pms_tokens = override_scratchpad_tokens
            x[0, pw_start:pw_end] = actual_pw_tokens
            x[0, pms_start:pms_end] = actual_pms_tokens
        else:
            actual_pw_tokens, actual_pms_tokens = pred_pw_tokens, pred_pms_tokens
            x[0, pw_start:pw_end] = pred_pw_tokens
            x[0, pms_start:pms_end] = pred_pms_tokens

        # --- ETAPA 2: CAMPO CAUSAL POR COLUMNAS (DERECHA A IZQUIERDA) ---
        # 3.1. Predecir Unidades (offset +2)
        logits_units = self.forward_backbone(x, t_noise, stage_id=1, attn_mask=attn_mask)
        prob_w_u = F.softmax(logits_units[0, watt_start + 2], dim=-1).max().item()
        prob_ms_u = F.softmax(logits_units[0, ms_start + 2], dim=-1).max().item()
        pred_w_u = logits_units[0, watt_start + 2].argmax(dim=-1)
        pred_ms_u = logits_units[0, ms_start + 2].argmax(dim=-1)
        x[0, watt_start + 2] = pred_w_u
        x[0, ms_start + 2] = pred_ms_u

        # 3.2. Predecir Decenas (offset +1, con unidades fijadas como hechos del contexto)
        logits_tens = self.forward_backbone(x, t_noise, stage_id=2, attn_mask=attn_mask)
        prob_w_t = F.softmax(logits_tens[0, watt_start + 1], dim=-1).max().item()
        prob_ms_t = F.softmax(logits_tens[0, ms_start + 1], dim=-1).max().item()
        pred_w_t = logits_tens[0, watt_start + 1].argmax(dim=-1)
        pred_ms_t = logits_tens[0, ms_start + 1].argmax(dim=-1)
        x[0, watt_start + 1] = pred_w_t
        x[0, ms_start + 1] = pred_ms_t

        # 3.3. Predecir Centenas (offset +0, con decenas y unidades fijadas en contexto)
        logits_hundreds = self.forward_backbone(x, t_noise, stage_id=3, attn_mask=attn_mask)
        prob_w_h = F.softmax(logits_hundreds[0, watt_start], dim=-1).max().item()
        prob_ms_h = F.softmax(logits_hundreds[0, ms_start], dim=-1).max().item()
        pred_w_h = logits_hundreds[0, watt_start].argmax(dim=-1)
        pred_ms_h = logits_hundreds[0, ms_start].argmax(dim=-1)
        x[0, watt_start] = pred_w_h
        x[0, ms_start] = pred_ms_h

        pred_watt_tokens = torch.stack([pred_w_h, pred_w_t, pred_w_u])
        pred_ms_tokens = torch.stack([pred_ms_h, pred_ms_t, pred_ms_u])

        # Decodificación final de caracteres ASCII
        pw_str = "".join(chr(int(t)) for t in actual_pw_tokens)
        pms_str = "".join(chr(int(t)) for t in actual_pms_tokens)
        pred_pw_raw = "".join(chr(int(t)) for t in pred_pw_tokens)
        pred_pms_raw = "".join(chr(int(t)) for t in pred_pms_tokens)
        watt_str = "".join(chr(int(t)) for t in pred_watt_tokens)
        ms_str = "".join(chr(int(t)) for t in pred_ms_tokens)

        full_output_str = f"pw: {pw_str}\npms: {pms_str}\nwatt: {watt_str}\nms: {ms_str}\n"

        pw_confs = probs_pw.max(dim=-1).values.tolist()
        pms_confs = probs_pms.max(dim=-1).values.tolist()
        watt_confs = [prob_w_h, prob_w_t, prob_w_u]
        ms_confs = [prob_ms_h, prob_ms_t, prob_ms_u]
        digit_confidences = pw_confs + pms_confs + watt_confs + ms_confs

        def _safe_int(s: str) -> int | None:
            try:
                return int(s)
            except (ValueError, TypeError):
                return None

        meta = {
            "pw": pw_str,
            "pms": pms_str,
            "pred_pw_raw": pred_pw_raw,
            "pred_pms_raw": pred_pms_raw,
            "watt": watt_str,
            "ms": ms_str,
            "pw_int": _safe_int(pw_str),
            "pms_int": _safe_int(pms_str),
            "pred_pw_raw_int": _safe_int(pred_pw_raw),
            "pred_pms_raw_int": _safe_int(pred_pms_raw),
            "watt_int": _safe_int(watt_str),
            "ms_int": _safe_int(ms_str),
            "digit_confidences": digit_confidences,
        }

        return full_output_str, meta
