# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
TRACELanguageModel: Modelo de Lenguaje Autorregresivo de Ultra-Baja Precisión
Combina MLA 100% Ternaria, FFN denso ternario (Squared-ReLU) y tied embeddings.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from trace.layers.norm import RMSNorm
from trace.layers.bitlinear import BitLinear
from trace.layers.ffn import TernaryExpert
from trace.lm.config import TRACELMConfig
from trace.lm.mla import MultiHeadLatentAttention


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
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_weight_absorption: bool = False,
        quantize_kv_int8: bool = False,
        add_sub_mode: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def build_causal_mask(self, seq_len: int, total_len: int, device: torch.device) -> Optional[torch.Tensor]:
        """Máscara causal compatible con Prefix Caching (past_len > 0 y seq_len >= 1)."""
        if seq_len == 1 and total_len == 1:
            return None
        past_len = total_len - seq_len
        mask = torch.zeros((seq_len, total_len), device=device)
        if seq_len > 1:
            causal_block = torch.full((seq_len, seq_len), float("-inf"), device=device)
            causal_block = torch.triu(causal_block, diagonal=1)
            mask[:, past_len:] = causal_block
        return mask.unsqueeze(0).unsqueeze(0) # [1, 1, seq_len, total_len]

    def configure_adamw_groups(
        self,
        bitlinear_weight_decay: float = 0.0,
        weight_decay: float = 0.01
    ) -> list:
        """Configuración de 3 grupos desacoplados de AdamW (estilo BitNet b1.58)."""
        bitlinear_param_ids = {id(mod.weight) for mod in self.modules() if isinstance(mod, BitLinear)}

        bitlinear_params = []
        dense_decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in bitlinear_param_ids:
                bitlinear_params.append(param)
            elif "emb" in name or param.dim() < 2:
                # Embeddings léxicos y biases/escalas 1D no decaen
                no_decay_params.append(param)
            else:
                dense_decay_params.append(param)

        return [
            {"name": "bitlinear_shadow", "params": bitlinear_params, "weight_decay": bitlinear_weight_decay},
            {"name": "dense_decay", "params": dense_decay_params, "weight_decay": weight_decay},
            {"name": "no_decay", "params": no_decay_params, "weight_decay": 0.0},
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[list] = None,
        use_weight_absorption: bool = False,
        quantize_kv_int8: bool = False,
        add_sub_mode: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], list]:
        B, S = input_ids.shape
        past_len = kv_caches[0][0].shape[1] if kv_caches is not None else 0
        total_len = past_len + S
        add_sub = self.cfg.use_add_sub_inference if add_sub_mode is None else add_sub_mode

        x = self.tok_emb(input_ids)
        attn_mask = self.build_causal_mask(S, total_len, x.device)

        new_caches = []
        for idx, layer in enumerate(self.layers):
            c_in = kv_caches[idx] if kv_caches is not None else None
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
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=self.cfg.pad_token_id,
            )

        return logits, loss, new_caches

    @staticmethod
    def _sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        """Helper para muestreo o selección greedy determinista."""
        if temperature <= 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        scaled_logits = logits / max(temperature, 1e-5)
        if top_k > 0:
            v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.shape[-1]))
            scaled_logits[scaled_logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
        quantize_kv_int8: bool = False,
    ) -> torch.Tensor:
        """Generación con prefill único y decodificación incremental token a token con kv_caches y absorción."""
        self.eval()
        if max_new_tokens <= 0:
            return input_ids

        eos_id = eos_token_id if eos_token_id is not None else self.cfg.eos_token_id
        curr_ids = input_ids.clone()
        if curr_ids.shape[1] >= self.cfg.max_seq_len:
            return curr_ids

        # 1. Prefill único
        logits, _, kv_caches = self(
            curr_ids,
            use_weight_absorption=True,
            quantize_kv_int8=quantize_kv_int8
        )
        next_token = self._sample_next_token(logits[:, -1, :], temperature, top_k)
        curr_ids = torch.cat([curr_ids, next_token], dim=1)

        if (next_token == eos_id).all():
            return curr_ids

        # 2. Decodificación incremental token a token (S = 1) con absorción de pesos
        for _ in range(max_new_tokens - 1):
            if curr_ids.shape[1] >= self.cfg.max_seq_len:
                break

            logits, _, kv_caches = self(
                next_token,
                kv_caches=kv_caches,
                use_weight_absorption=True,
                quantize_kv_int8=quantize_kv_int8
            )
            next_token = self._sample_next_token(logits[:, -1, :], temperature, top_k)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)

            if (next_token == eos_id).all():
                break

        return curr_ids
