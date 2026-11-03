#!/usr/bin/env python3
"""PLL-word-l2r scoring for the Persian whole-continuation analysis."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import score_continuation_yek_v553 as base  # noqa: E402


def _word_spans(text: str) -> List[Tuple[int, int]]:
    """Whitespace-based word spans. Treats punctuation attached to a word as part of that word."""
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _find_word_index(tok_start: int, tok_end: int, spans: List[Tuple[int, int]]) -> int | None:
    """Return index of word span overlapping token span."""
    if tok_end <= tok_start:
        return None
    for i, (ws, we) in enumerate(spans):
        if tok_start < we and tok_end > ws:
            return i
    return None


@dataclass
class PLLWordL2RScorer:
    model_name: str
    device: str | None = None

    def __post_init__(self) -> None:
        print(f"  Loading {self.model_name} with PLL-word-l2r scorer …", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError(
                f"{self.model_name} did not load a fast tokenizer. "
                "PLL-word-l2r needs offset mappings."
            )
        if self.tokenizer.mask_token_id is None:
            raise RuntimeError(f"{self.model_name} has no mask token.")

        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name).eval()
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        self.model.to(self.device)
        print(f"  Loaded on {self.device}.", flush=True)

    @torch.no_grad()
    def pll_len_norm(self, text: str) -> float:
        """
        Mean per-token log-probability under PLL-word-l2r.

        For token t in word w, mask token t and all later tokens of word w.
        """
        text = str(text)
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"]
        offsets = enc["offset_mapping"][0].tolist()

        spans = _word_spans(text)

        # Map each real token to a word index.
        token_to_word: Dict[int, int] = {}
        positions: List[int] = []
        for i, (start, end) in enumerate(offsets):
            # Skip special tokens and zero-length tokens.
            if end <= start:
                continue
            wid = _find_word_index(start, end, spans)
            if wid is None:
                continue
            token_to_word[i] = wid
            positions.append(i)

        if not positions:
            return float("nan")

        # For each target position, mask current token plus future tokens
        # belonging to the same word.
        bsz = len(positions)
        ids_batch = input_ids.unsqueeze(0).repeat(bsz, 1)
        att_batch = attention_mask.repeat(bsz, 1)

        for row_idx, target_pos in enumerate(positions):
            wid = token_to_word[target_pos]
            for pos in positions:
                if pos >= target_pos and token_to_word.get(pos) == wid:
                    ids_batch[row_idx, pos] = self.tokenizer.mask_token_id

        batch = {
            "input_ids": ids_batch.to(self.device),
            "attention_mask": att_batch.to(self.device),
        }

        # Include token_type_ids for BERT-like models if available.
        if "token_type_ids" in enc:
            batch["token_type_ids"] = enc["token_type_ids"].repeat(bsz, 1).to(self.device)

        logits = self.model(**batch).logits
        log_prob = torch.log_softmax(logits, dim=-1)

        token_logps = []
        for row_idx, target_pos in enumerate(positions):
            gold_id = int(input_ids[target_pos].item())
            token_logps.append(float(log_prob[row_idx, target_pos, gold_id].item()))

        return float(np.mean(token_logps))


# Monkey-patch the original scorer and output paths.
base.PLLScorer = PLLWordL2RScorer
PRIMARY_OUT = ROOT / "results" / "primary_pll_word_l2r" / "whole_continuation"

base.OUT_CSV = PRIMARY_OUT / "continuation_yek_v5_5_3_pll_word_l2r_scores.csv"
base.OUT_JSON = PRIMARY_OUT / "continuation_yek_v5_5_3_pll_word_l2r_summary.json"
base.OUT_REPORT = PRIMARY_OUT / "continuation_yek_v5_5_3_pll_word_l2r_report.md"

if __name__ == "__main__":
    if not hasattr(base, "main"):
        raise RuntimeError("score_continuation_yek_v553.py has no main() function.")
    base.main()
