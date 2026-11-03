#!/usr/bin/env python3
"""PLL-word-l2r scoring for the Persian span analysis."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import score_yek_span_controls as base  # noqa: E402


def _word_spans(text: str) -> List[Tuple[int, int]]:
    """
    Whitespace-based word spans.

    Uses whitespace-delimited spans. ZWNJ handling is performed by the span
    matcher before these offsets are used.
    """
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", str(text))]


def _find_word_index(tok_start: int, tok_end: int, spans: List[Tuple[int, int]]) -> Optional[int]:
    """Return the index of the whitespace word overlapping this token span."""
    if tok_end <= tok_start:
        return None
    for i, (ws, we) in enumerate(spans):
        if tok_start < we and tok_end > ws:
            return i
    return None


class PLLWordL2RSpanScorer:
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if not self.tokenizer.is_fast:
            raise RuntimeError(f"Tokenizer for {model_name} is not fast; offset mapping is required.")
        if self.tokenizer.mask_token_id is None:
            raise RuntimeError(f"Tokenizer for {model_name} has no mask token.")

        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

        if device is None or device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score_span(self, text: str, char_start: int, char_end: int) -> Dict[str, object]:
        text = str(text)

        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=False,
        )
        input_ids = enc["input_ids"][0]
        offsets = enc["offset_mapping"][0].tolist()

        spans = _word_spans(text)

        token_to_word: Dict[int, int] = {}
        target_token_indices: List[int] = []

        for i, (s, e) in enumerate(offsets):
            # Special tokens usually have (0, 0).
            if s == e:
                continue

            wid = _find_word_index(s, e, spans)
            if wid is None:
                continue
            token_to_word[i] = wid

            # Include tokens overlapping the requested target span.
            if e > char_start and s < char_end:
                target_token_indices.append(i)

        if not target_token_indices:
            return {
                "score_mean": np.nan,
                "score_sum": np.nan,
                "target_token_count": 0,
                "target_token_ids": "",
                "target_tokens": "",
                "score_ok": False,
                "score_notes": "no_target_tokens_from_offsets",
            }

        logps = []
        token_strs = []
        target_ids = []

        for token_idx in target_token_indices:
            original_id = int(input_ids[token_idx].item())
            wid = token_to_word.get(token_idx)

            masked = input_ids.clone()

            # PLL-word-l2r: mask current token and all future tokens in same word.
            # Future tokens from the same word may extend beyond the requested
            # character span if the span cuts through a word.
            for pos, pos_wid in token_to_word.items():
                if wid is not None and pos_wid == wid and pos >= token_idx:
                    masked[pos] = self.tokenizer.mask_token_id

            batch = {
                "input_ids": masked.unsqueeze(0).to(self.device),
                "attention_mask": enc["attention_mask"].to(self.device),
            }
            if "token_type_ids" in enc:
                batch["token_type_ids"] = enc["token_type_ids"].to(self.device)

            logits = self.model(**batch).logits[0, token_idx]
            logprob = torch.log_softmax(logits, dim=-1)[original_id].item()

            logps.append(logprob)
            target_ids.append(str(original_id))
            token_strs.append(self.tokenizer.convert_ids_to_tokens([original_id])[0])

        return {
            "score_mean": float(np.mean(logps)),
            "score_sum": float(np.sum(logps)),
            "target_token_count": len(logps),
            "target_token_ids": " ".join(target_ids),
            "target_tokens": " ".join(token_strs),
            "score_ok": True,
            "score_notes": "pll_word_l2r",
        }


base.PLLSpanScorer = PLLWordL2RSpanScorer

if __name__ == "__main__":
    if "--outdir" not in sys.argv:
        sys.argv.extend(["--outdir", str(ROOT / "results" / "primary_pll_word_l2r" / "span_controls")])
    base.main()
