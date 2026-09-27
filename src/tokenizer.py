"""Byte-level BPE tokenizer, built from tokenizer.json alone.

Pipeline (encode):  special-token split -> NFC normalize -> regex pre-split -> bytes -> stand-in chars
                    -> BPE merges by priority -> vocab lookup.   Decode runs the last steps backwards.
"""
import json
import unicodedata

import regex  # third-party engine: needed for \p{L} / \p{N} classes that the stdlib `re` lacks


def build_byte_alphabet() -> tuple[dict[int, str], dict[str, int]]:
    """The 256 atoms: every byte value 0..255 gets a printable stand-in character.
    Printable bytes stand for themselves; the rest are shifted up to 256+ so they stay visible
    (space -> 'Ġ', newline -> 'Ċ'). This is GPT-2's bytes_to_unicode, which Qwen inherits."""
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    byte_to_char: dict[int, str] = {}
    nxt = 256
    for b in range(256):
        if b in printable:
            byte_to_char[b] = chr(b)
        else:
            byte_to_char[b] = chr(nxt)
            nxt += 1
    char_to_byte = {c: b for b, c in byte_to_char.items()}
    return byte_to_char, char_to_byte


class Tokenizer:
    def __init__(self, path: str):
        spec = json.load(open(path, encoding="utf-8"))
        model = spec["model"]

        self.vocab: dict[str, int] = model["vocab"]                       # token string -> id
        self.id_to_token: dict[int, str] = {i: t for t, i in self.vocab.items()}

        # merges are either "a b" strings or [a, b] pairs depending on file version; rank = position
        self.merge_rank: dict[tuple[str, str], int] = {}
        for rank, m in enumerate(model["merges"]):
            a, b = m.split(" ") if isinstance(m, str) else m
            self.merge_rank[(a, b)] = rank

        self.special: dict[str, int] = {t["content"]: t["id"] for t in spec["added_tokens"]}
        self.id_to_token.update({i: t for t, i in self.special.items()})
        self.special_re = regex.compile("(" + "|".join(regex.escape(s) for s in self.special) + ")")

        # The pre-tokenizer regex, taken VERBATIM from the file (the `regex` engine supports (?i:...)).
        # Note: for Qwen3.5 checkpoints, transformers' AutoTokenizer resolves to Qwen2Tokenizer (per the checkpoint's
        # tokenizer_config.json) and substitutes the older Qwen2 regex, which lacks \p{M}; it then splits combining
        # marks (Hindi, Tamil, Thai, Arabic) differently from this file. The file and transformers' own
        # Qwen3_5Tokenizer agree with each other, and this engine follows them.
        pattern = spec["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
        self.pre_re = regex.compile(pattern)

        self.byte_to_char, self.char_to_byte = build_byte_alphabet()

    # ---------- Stage 3: BPE merging for ONE chunk ----------
    def _bpe(self, chunk: str) -> list[str]:
        symbols = list(chunk)                                  # start as single stand-in characters
        while len(symbols) > 1:
            # find the adjacent pair with the best (lowest) merge rank
            best, best_rank = None, None
            for pair in zip(symbols, symbols[1:]):
                r = self.merge_rank.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = pair, r
            if best is None:
                break                                          # nothing left to merge
            # glue every occurrence of that pair
            merged, out, i = best[0] + best[1], [], 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best:
                    out.append(merged)
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            symbols = out
        return symbols

    # ---------- Encode: text -> ids ----------
    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for part in self.special_re.split(text):                          # Stage 0: peel off specials
            if not part:
                continue
            if part in self.special:
                ids.append(self.special[part])
                continue
            part = unicodedata.normalize("NFC", part)                    # Stage 1
            for chunk in self.pre_re.findall(part):                       # Stage 2: word-like chunks
                mapped = "".join(self.byte_to_char[b] for b in chunk.encode("utf-8"))  # Stage 2b
                for piece in self._bpe(mapped):                           # Stage 3
                    ids.append(self.vocab[piece])                         # vocab lookup
        return ids

    # ---------- Decode: ids -> text ----------
    def decode(self, ids: list[int]) -> str:
        out = bytearray()
        for i in ids:
            tok = self.id_to_token[i]
            if tok in self.special:
                out += tok.encode("utf-8")
            else:
                out += bytes(self.char_to_byte[c] for c in tok)           # stand-in char -> byte
        return out.decode("utf-8", errors="replace")

    def vocab_size(self) -> int:
        return len(self.id_to_token)
