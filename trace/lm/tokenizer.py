# Basic-Lite
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""
Tokenizador BPE de 4096 Tokens con Pre-Segmentador Determinista por Bytes
Garantiza paridad bit a bit con el motor nativo en C (csrc/lm/trace_tokenizer.c).
"""

import os
import struct
from collections import Counter
from typing import List, Tuple, Dict, Iterator


VOCAB_SIZE = 4096
SPECIAL_PAD = 256
SPECIAL_MASK = 257
SPECIAL_BOS = 258
SPECIAL_EOS = 259
FIRST_MERGE = 260
NUM_MERGES = VOCAB_SIZE - FIRST_MERGE


def _classify_cp(cp: int) -> int:
    """Clasificación determinista de codepoint UTF-8 sin regex."""
    if 48 <= cp <= 57:  # '0'..'9'
        return 0  # CAT_DIGIT
    if cp == 32 or cp == 9:  # ' ', '\t'
        return 1  # CAT_SPACE
    if cp == 13 or cp == 10:  # '\r', '\n'
        return 2  # CAT_NEWLINE
    if (97 <= cp <= 122) or (65 <= cp <= 90) or cp == 95:  # 'a'..'z', 'A'..'Z', '_'
        return 3  # CAT_WORD
    if (0xC0 <= cp <= 0xD6) or (0xD8 <= cp <= 0xF6) or (0xF8 <= cp <= 0x024F):
        return 3  # CAT_WORD (Latin-1 y Latin Extended acentuadas)
    if 33 <= cp <= 126:
        return 4  # CAT_PUNCT
    return 5  # CAT_OTHER


def _decode_utf8_char(raw: bytes, offset: int) -> Tuple[int, int]:
    """Decodifica un punto de código UTF-8 y su longitud en bytes."""
    b0 = raw[offset]
    rem = len(raw) - offset
    if b0 < 0x80:
        return b0, 1
    if (b0 & 0xE0) == 0xC0 and rem >= 2 and (raw[offset + 1] & 0xC0) == 0x80:
        return ((b0 & 0x1F) << 6) | (raw[offset + 1] & 0x3F), 2
    if (b0 & 0xF0) == 0xE0 and rem >= 3 and (raw[offset + 1] & 0xC0) == 0x80 and (raw[offset + 2] & 0xC0) == 0x80:
        return ((b0 & 0x0F) << 12) | ((raw[offset + 1] & 0x3F) << 6) | (raw[offset + 2] & 0x3F), 3
    if (b0 & 0xF8) == 0xF0 and rem >= 4 and (raw[offset + 1] & 0xC0) == 0x80 and (raw[offset + 2] & 0xC0) == 0x80 and (raw[offset + 3] & 0xC0) == 0x80:
        return ((b0 & 0x07) << 18) | ((raw[offset + 1] & 0x3F) << 12) | ((raw[offset + 2] & 0x3F) << 6) | (raw[offset + 3] & 0x3F), 4
    return b0, 1


def pre_tokenize(text_bytes: bytes) -> Iterator[bytes]:
    """Pre-segmentador determinista idéntico a classify_cp en C."""
    n = len(text_bytes)
    i = 0
    while i < n:
        cp, cp_len = _decode_utf8_char(text_bytes, i)
        cat = _classify_cp(cp)

        # Regla 1: Dígitos aislados siempre en 1 byte
        if cat == 0:
            yield text_bytes[i:i + 1]
            i += 1
            continue

        # Regla 2: Emojis y multibyte no-palabra aislados individualmente
        if cat == 5:
            yield text_bytes[i:i + cp_len]
            i += cp_len
            continue

        # Prefijo de espacio único en palabras (' ' + WORD)
        if cat == 1 and cp == 32 and cp_len == 1:
            if i + 1 < n:
                next_cp, next_len = _decode_utf8_char(text_bytes, i + 1)
                if _classify_cp(next_cp) == 3:  # CAT_WORD
                    j = i + 1 + next_len
                    while j < n:
                        w_cp, w_len = _decode_utf8_char(text_bytes, j)
                        if _classify_cp(w_cp) != 3:
                            break
                        j += w_len
                    yield text_bytes[i:j]
                    i = j
                    continue

        # Corridas regulares de la misma categoría
        j = i + cp_len
        while j < n:
            next_cp, next_len = _decode_utf8_char(text_bytes, j)
            if _classify_cp(next_cp) != cat:
                break
            j += next_len

        yield text_bytes[i:j]
        i = j


class TRACETokenizer:
    def __init__(self, bin_path: str = None):
        self.vocab_size = VOCAB_SIZE
        self.num_merges = NUM_MERGES
        self.merges: Dict[Tuple[int, int], int] = {}
        self.pieces: List[bytes] = [bytes([i]) for i in range(256)]
        self.pieces.extend([b"<PAD>", b"<MASK>", b"<BOS>", b"<EOS>"])

        if bin_path and os.path.exists(bin_path):
            self.load(bin_path)

    def load(self, bin_path: str) -> None:
        with open(bin_path, "rb") as f:
            hdr_bytes = f.read(32)
            magic, ver, v_sz, n_m, pad, mask, bos, eos = struct.unpack("<8I", hdr_bytes)
            assert magic == 0x4B545254, "Magic inválido"
            assert ver == 1, "Versión de formato inválida"
            assert v_sz == VOCAB_SIZE, "Vocab size incorrecto"

            merges_bytes = f.read(n_m * 8)
            merges_data = struct.unpack(f"<{n_m * 2}I", merges_bytes)
            self.merges.clear()
            for i in range(n_m):
                left = merges_data[2 * i]
                right = merges_data[2 * i + 1]
                self.merges[(left, right)] = FIRST_MERGE + i
            self.num_merges = n_m

            self.pieces = []
            for _ in range(VOCAB_SIZE):
                (p_len,) = struct.unpack("<H", f.read(2))
                self.pieces.append(f.read(p_len))

    def save(self, bin_path: str) -> None:
        assert len(self.pieces) == VOCAB_SIZE, f"pieces debe tener longitud {VOCAB_SIZE}"
        sorted_merges = sorted(self.merges.items(), key=lambda item: item[1])
        num_m = len(sorted_merges)
        with open(bin_path, "wb") as f:
            f.write(struct.pack(
                "<8I",
                0x4B545254, 1, VOCAB_SIZE, num_m,
                SPECIAL_PAD, SPECIAL_MASK, SPECIAL_BOS, SPECIAL_EOS
            ))
            for (left, right), token_id in sorted_merges:
                f.write(struct.pack("<2I", left, right))
            for p in self.pieces:
                f.write(struct.pack("<H", len(p)))
                f.write(p)

    @classmethod
    def train(
        cls,
        texts: List[str],
        target_vocab_size: int = VOCAB_SIZE,
        min_freq: int = 2
    ) -> "TRACETokenizer":
        """Entrena BPE de forma 100% determinista con actualización incremental de frecuencias."""
        target_merges = target_vocab_size - FIRST_MERGE
        assert target_merges > 0, "target_vocab_size debe ser > 260"

        # 1. Pre-tokenizar todo el corpus y contar secuencias únicas
        raw_counts: Dict[Tuple[int, ...], int] = Counter()
        for text in texts:
            for seg in pre_tokenize(text.encode("utf-8")):
                if len(seg) > 1:
                    raw_counts[tuple(seg)] += 1

        seq_list: List[List[int]] = [list(seq) for seq in raw_counts.keys()]
        freq_list: List[int] = list(raw_counts.values())

        # Mapa invertido: par -> set de índices de secuencias que lo contienen
        pair_to_seqs: Dict[Tuple[int, int], set] = {}
        pair_counts: Dict[Tuple[int, int], int] = Counter()

        for s_idx, (seq, freq) in enumerate(zip(seq_list, freq_list)):
            for i in range(len(seq) - 1):
                p = (seq[i], seq[i + 1])
                pair_counts[p] += freq
                if p not in pair_to_seqs:
                    pair_to_seqs[p] = set()
                pair_to_seqs[p].add(s_idx)

        merges_list: List[Tuple[int, int]] = []
        pieces: List[bytes] = [bytes([i]) for i in range(256)]
        pieces.extend([b"<PAD>", b"<MASK>", b"<BOS>", b"<EOS>"])

        # 2. Bucle determinista de fusiones con actualización incremental
        while len(merges_list) < target_merges:
            if not pair_counts:
                break

            # Desempate determinista: mayor frecuencia, luego menor ID lexicográfico
            best_pair = min(pair_counts.keys(), key=lambda p: (-pair_counts[p], p[0], p[1]))
            if pair_counts[best_pair] < min_freq:
                break

            # Aserción contractual: NUNCA fusionar dígitos ASCII
            assert not (48 <= best_pair[0] <= 57 or 48 <= best_pair[1] <= 57), (
                f"Violación: se intentó fusionar dígito ASCII: {best_pair}"
            )

            new_id = FIRST_MERGE + len(merges_list)
            merges_list.append(best_pair)
            pieces.append(pieces[best_pair[0]] + pieces[best_pair[1]])

            affected_seq_indices = list(pair_to_seqs.get(best_pair, set()))
            p0, p1 = best_pair

            for s_idx in affected_seq_indices:
                seq = seq_list[s_idx]
                freq = freq_list[s_idx]

                # Quitar pares obsoletos de la secuencia anterior
                for i in range(len(seq) - 1):
                    old_p = (seq[i], seq[i + 1])
                    pair_counts[old_p] -= freq
                    if pair_counts[old_p] <= 0:
                        pair_counts.pop(old_p, None)
                    if old_p in pair_to_seqs:
                        pair_to_seqs[old_p].discard(s_idx)
                        if not pair_to_seqs[old_p]:
                            del pair_to_seqs[old_p]

                # Reemplazar ocurrencias del par fusionado
                new_seq = []
                idx = 0
                n = len(seq)
                while idx < n:
                    if idx < n - 1 and seq[idx] == p0 and seq[idx + 1] == p1:
                        new_seq.append(new_id)
                        idx += 2
                    else:
                        new_seq.append(seq[idx])
                        idx += 1
                seq_list[s_idx] = new_seq

                # Agregar nuevos pares resultantes de la fusión
                for i in range(len(new_seq) - 1):
                    new_p = (new_seq[i], new_seq[i + 1])
                    pair_counts[new_p] = pair_counts.get(new_p, 0) + freq
                    if new_p not in pair_to_seqs:
                        pair_to_seqs[new_p] = set()
                    pair_to_seqs[new_p].add(s_idx)

        while len(pieces) < VOCAB_SIZE:
            pieces.append(b"")

        tok = cls()
        tok.merges = {pair: FIRST_MERGE + i for i, pair in enumerate(merges_list)}
        tok.pieces = pieces
        tok.num_merges = len(merges_list)
        return tok

    def _merge_segment(self, seg_bytes: bytes) -> List[int]:
        if not seg_bytes:
            return []
        if len(seg_bytes) == 1:
            return [seg_bytes[0]]

        tokens = list(seg_bytes)
        while len(tokens) >= 2:
            best_rank = 10**9
            best_pair = None
            best_idx = -1

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merges:
                    rank = self.merges[pair] - FIRST_MERGE
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair
                        best_idx = i

            if best_idx == -1:
                break

            merged_id = self.merges[best_pair]
            tokens = tokens[:best_idx] + [merged_id] + tokens[best_idx + 2:]

        return tokens

    def encode(self, text: str) -> List[int]:
        raw = text.encode("utf-8")
        out: List[int] = []
        for seg in pre_tokenize(raw):
            out.extend(self._merge_segment(seg))
        return out

    def decode(self, tokens: List[int]) -> str:
        out_bytes = bytearray()
        for t in tokens:
            if 0 <= t < len(self.pieces):
                if t in (SPECIAL_PAD, SPECIAL_MASK, SPECIAL_BOS, SPECIAL_EOS):
                    continue
                out_bytes.extend(self.pieces[t])
        return out_bytes.decode("utf-8", errors="replace")
