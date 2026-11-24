# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available

"""
TRACELanguageModel: Modelo de Lenguaje Autorregresivo de Ultra-Baja Precisión.
Combina MLA 100% Ternaria, FFN denso ternario (Squared-ReLU), tied embeddings,
soporte Continual QAT y gradient checkpointing opcional.
"""

from trace.layers.bitlinear import BitLinear
from trace.layers.ffn import TernaryExpert
from trace.layers.norm import RMSNorm
from trace.lm.config import TRACELMConfig
from trace.lm.mla import MultiHeadLatentAttention

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt_utils


class TRACELMBlock(nn.Module):
    def __init__(self, cfg: TRACELMConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.eps)
        self.attn = MultiHeadLatentAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, eps=cfg.eps)
        self.ffn = TernaryExpert(cfg.d_model, cfg.d_ff, eps=cfg.eps)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_weight_absorption: bool = False,
        quantize_kv_int8: bool = False,
        add_sub_mode: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h, new_cache = self.attn(
            self.attn_norm(x),
            attn_mask=attn_mask,
            kv_cache=kv_cache,
            use_weight_absorption=use_weight_absorption,
            quantize_kv_int8=quantize_kv_int8,
            add_sub_mode=add_sub_mode,
        )
        x = x + h
        x = x + self.ffn(self.ffn_norm(x), add_sub_mode=add_sub_mode)
        return x, new_cache


class TRACELanguageModel(nn.Module):
    def __init__(self, cfg: TRACELMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([TRACELMBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        self._apply_quant_config()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _apply_quant_config(self) -> None:
        """Propaga weight_scale_type y quantize_weights_enabled a todos los BitLinear (MLA + FFN)."""
        for m in self.modules():
            if isinstance(m, BitLinear):
                m.quantize_weights_enabled = self.cfg.quantize_weights_enabled
                m.weight_scale_type = self.cfg.weight_scale_type

    def set_quantize_weights(self, enabled: bool) -> None:
        """Activa o desactiva la cuantización ternaria W1.58 (soporte Continual QAT)."""
        self.cfg.quantize_weights_enabled = enabled
        for m in self.modules():
            if isinstance(m, BitLinear):
                m.quantize_weights_enabled = enabled
                if m.cached_w_eff.numel() > 0:
                    m.cached_w_eff = torch.empty(0, device=m.weight.device)

    def build_causal_mask(
        self,
        seq_len: int,
        total_len: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Máscara causal compatible con Prefix Caching (past_len > 0 y seq_len >= 1)."""
        if seq_len == 1 and total_len == 1:
            return None
        past_len = total_len - seq_len
        mask = torch.zeros((seq_len, total_len), device=device)
        if seq_len > 1:
            causal_block = torch.full((seq_len, seq_len), float("-inf"), device=device)
            causal_block = torch.triu(causal_block, diagonal=1)
            mask[:, past_len:] = causal_block
        return mask.unsqueeze(0).unsqueeze(0)

    def configure_adamw_groups(
        self,
        bitlinear_weight_decay: float = 0.0,
        weight_decay: float = 0.01,
    ) -> list[dict]:
        """Configuración de 3 grupos desacoplados de AdamW (estilo BitNet b1.58)."""
        bitlinear_ids = {id(mod.weight) for mod in self.modules() if isinstance(mod, BitLinear)}

        bitlinear_params = []
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in bitlinear_ids:
                bitlinear_params.append(param)
            elif "emb" in name or param.dim() < 2:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        groups = []
        if bitlinear_params:
            groups.append({"name": "bitlinear_shadow", "params": bitlinear_params, "weight_decay": bitlinear_weight_decay})
        if decay_params:
            groups.append({"name": "dense_decay", "params": decay_params, "weight_decay": weight_decay})
        if no_decay_params:
            groups.append({"name": "no_decay", "params": no_decay_params, "weight_decay": 0.0})
        return groups

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_caches: list | None = None,
        use_weight_absorption: bool = False,
        quantize_kv_int8: bool = False,
        add_sub_mode: bool | None = None,
        use_grad_checkpoint: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list]:
        _, S = input_ids.shape
        past_len = kv_caches[0][0].shape[1] if kv_caches is not None else 0
        total_len = past_len + S
        add_sub = self.cfg.use_add_sub_inference if add_sub_mode is None else add_sub_mode

        x = self.tok_emb(input_ids)

        # Sin caché y sin absorción -> attn_mask=None habilita SDPA is_causal=True (FlashAttention-2)
        if kv_caches is None and not use_weight_absorption:
            attn_mask = None
        else:
            attn_mask = self.build_causal_mask(S, total_len, x.device)

        new_caches = []
        for idx, layer in enumerate(self.layers):
            c_in = kv_caches[idx] if kv_caches is not None else None
            if self.training and use_grad_checkpoint and kv_caches is None:

                def _ckpt_fn(inp: torch.Tensor, _layer: TRACELMBlock = layer) -> torch.Tensor:
                    out, _ = _layer(
                        inp,
                        attn_mask=attn_mask,
                        kv_cache=None,
                        use_weight_absorption=use_weight_absorption,
                        quantize_kv_int8=quantize_kv_int8,
                        add_sub_mode=add_sub,
                    )
                    return out

                x = ckpt_utils.checkpoint(_ckpt_fn, x, use_reentrant=False)
                new_caches.append(None)
            else:
                x, c_out = layer(
                    x,
                    attn_mask=attn_mask,
                    kv_cache=c_in,
                    use_weight_absorption=use_weight_absorption,
                    quantize_kv_int8=quantize_kv_int8,
                    add_sub_mode=add_sub,
                )
                new_caches.append(c_out)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=self.cfg.pad_token_id,
            )

        return logits, loss, new_caches

    @staticmethod
    def _sample_next_token(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
        if temperature == 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        scaled = logits / max(temperature, 1e-5)
        if top_k > 0:
            v, _ = torch.topk(scaled, min(top_k, scaled.shape[-1]))
            scaled = scaled.masked_fill(scaled < v[:, [-1]], float("-inf"))
        probs = F.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int = 50,
        eos_token_id: int | None = None,
        quantize_kv_int8: bool = False,
    ) -> torch.Tensor:
        self.eval()
        curr_ids = input_ids.clone()
        if max_new_tokens <= 0 or curr_ids.shape[1] >= self.cfg.max_seq_len:
            return curr_ids

        eos_id = eos_token_id if eos_token_id is not None else self.cfg.eos_token_id

        logits, _, kv_caches = self(
            curr_ids,
            use_weight_absorption=True,
            quantize_kv_int8=quantize_kv_int8,
        )
        next_token = self._sample_next_token(logits[:, -1, :], temperature, top_k)
        curr_ids = torch.cat([curr_ids, next_token], dim=1)
        if (next_token == eos_id).all():
            return curr_ids

        for _ in range(max_new_tokens - 1):
            if curr_ids.shape[1] >= self.cfg.max_seq_len:
                break
            logits, _, kv_caches = self(
                next_token,
                kv_caches=kv_caches,
                use_weight_absorption=True,
                quantize_kv_int8=quantize_kv_int8,
            )
            next_token = self._sample_next_token(logits[:, -1, :], temperature, top_k)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            if (next_token == eos_id).all():
                break

        return curr_ids
