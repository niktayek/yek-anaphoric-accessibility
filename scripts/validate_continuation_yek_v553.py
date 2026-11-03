#!/usr/bin/env python3
"""Validate the frozen Persian stimulus CSV.

Writes ``results/continuation_yek_v5_5_3_validation_summary.json``.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "stimuli" / "continuation_yek_v5_5_3_same_np_smoothing_48.csv"

# --- Input existence check ---
def ensure_v553_csv_exists() -> None:
    """Raise ``FileNotFoundError`` if the input CSV is missing."""
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV.relative_to(ROOT)}")
OUT_JSON = ROOT / "results" / "continuation_yek_v5_5_3_validation_summary.json"

EXPECTED_COLUMNS = [
    "item_id",
    "condition",
    "subject",
    "object",
    "verb",
    "context",
    "continuation_subject_continuing",
    "continuation_object_anaphoric",
    "object_anaphor_type",
    "object_domain",
    "target_span",
    "notes",
]


CONTENT_WORD_STOPWORDS = {
    "را", "و", "با", "به", "از", "در", "که", "این", "آن", "همان", "یک",
    "هم", "هر", "برای", "بعد", "بعدا", "بعداً", "سپس", "کمی", "چند", "خیلی",
    "زود", "روز", "شب", "تا", "بود", "شد", "کرد", "می", "ها", "های",
    "تر", "تره", "ترش", "داشت", "ماند", "گرفت", "داد", "رفت", "آمد",
}

PUNCT_RE = re.compile(r"[\.،؛:!؟,\-ـ()\[\]{}\"']+")


def normalize_space(text: str) -> str:
    text = text.replace("\u200c", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def has_standalone_ra(text: str) -> bool:
    # Standalone token را, including punctuation boundaries.
    return re.search(r"(^|\s)را(?=\s|[\.,،؛!?]|$)", text) is not None


def content_words(text: str) -> List[str]:
    cleaned = PUNCT_RE.sub(" ", normalize_space(text))
    words = []
    for w in cleaned.split():
        w = w.strip()
        if not w or w in CONTENT_WORD_STOPWORDS:
            continue
        if len(w) <= 1:
            continue
        words.append(w)
    return words


def load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    # Guard against Apple Numbers / Excel package files that were renamed to .csv.
    # Real UTF-8 CSV files must not start with the ZIP magic bytes PK.
    with path.open("rb") as raw_f:
        first_bytes = raw_f.read(4)
    if first_bytes.startswith(b"PK"):
        raise ValueError(
            f"Input file is not a real CSV: {path}\n"
            "It appears to be a ZIP-based spreadsheet/package file, probably Apple Numbers or XLSX, "
            "renamed with a .csv extension. Export it as UTF-8 CSV and rerun the validator."
        )

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != EXPECTED_COLUMNS:
                raise ValueError(
                    f"Unexpected columns. Expected {EXPECTED_COLUMNS}, got {reader.fieldnames}"
                )
            rows = list(reader)
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Input file is not readable as UTF-8 CSV: {path}\n"
            "Export the spreadsheet as UTF-8 CSV instead of renaming another file type to .csv."
        ) from exc

    return rows


def expected_context(subject: str, obj: str, verb: str, condition: str) -> str:
    if condition == "bare":
        return f"{subject} {obj} {verb}."
    if condition == "yek":
        return f"{subject} یک {obj} {verb}."
    return ""


def run_checks(rows: List[Dict[str, str]]) -> Tuple[Dict, List[str]]:
    issues: List[str] = []
    summary: Dict = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(INPUT_CSV.relative_to(ROOT)),
        "output_json": str(OUT_JSON.relative_to(ROOT)),
        "total_rows": len(rows),
        "unique_item_ids": 0,
        "checks": {},
        "counts": {},
        "review_checks": {},
    }

    by_item: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_item[r["item_id"]].append(r)

    unique_items = len(by_item)
    summary["unique_item_ids"] = unique_items

    # 1) row/item counts
    c1_ok = len(rows) == 96 and unique_items == 48
    summary["checks"]["row_and_item_count"] = {
        "pass": c1_ok,
        "expected_rows": 96,
        "actual_rows": len(rows),
        "expected_unique_item_ids": 48,
        "actual_unique_item_ids": unique_items,
    }

    # 2) exactly one bare and one yek per item
    bad_pairing = []
    for item_id, grp in by_item.items():
        conds = Counter(r["condition"] for r in grp)
        if conds.get("bare", 0) != 1 or conds.get("yek", 0) != 1 or len(grp) != 2:
            bad_pairing.append({"item_id": item_id, "condition_counts": dict(conds)})
    summary["checks"]["pairing_one_bare_one_yek"] = {
        "pass": len(bad_pairing) == 0,
        "violations": bad_pairing,
    }

    # 3) bare/yek contexts differ only by insertion of یک before object NP
    context_pair_violations = []
    for item_id, grp in by_item.items():
        if len(grp) != 2:
            continue
        rows_by_cond = {r["condition"]: r for r in grp}
        if "bare" not in rows_by_cond or "yek" not in rows_by_cond:
            continue
        b = rows_by_cond["bare"]
        y = rows_by_cond["yek"]

        b_expected = expected_context(b["subject"], b["object"], b["verb"], "bare")
        y_expected = expected_context(y["subject"], y["object"], y["verb"], "yek")

        b_ok = normalize_space(b["context"]) == normalize_space(b_expected)
        y_ok = normalize_space(y["context"]) == normalize_space(y_expected)

        if not b_ok or not y_ok:
            context_pair_violations.append(
                {
                    "item_id": item_id,
                    "bare_context": b["context"],
                    "bare_expected": b_expected,
                    "yek_context": y["context"],
                    "yek_expected": y_expected,
                }
            )

    summary["checks"]["context_only_yek_insertion"] = {
        "pass": len(context_pair_violations) == 0,
        "violations": context_pair_violations,
    }

    # 3b) paired rows must match on all design fields except condition/context/notes
    pair_field_mismatches = []
    paired_fields = [
        "subject",
        "object",
        "verb",
        "continuation_subject_continuing",
        "continuation_object_anaphoric",
        "object_anaphor_type",
        "object_domain",
        "target_span",
    ]
    for item_id, grp in by_item.items():
        if len(grp) != 2:
            continue
        rows_by_cond = {r["condition"]: r for r in grp}
        if "bare" not in rows_by_cond or "yek" not in rows_by_cond:
            continue
        b = rows_by_cond["bare"]
        y = rows_by_cond["yek"]
        mismatched_fields = []
        for field in paired_fields:
            if normalize_space(b[field]) != normalize_space(y[field]):
                mismatched_fields.append(
                    {
                        "field": field,
                        "bare_value": b[field],
                        "yek_value": y[field],
                    }
                )
        if mismatched_fields:
            pair_field_mismatches.append(
                {
                    "item_id": item_id,
                    "mismatched_fields": mismatched_fields,
                }
            )

    summary["checks"]["paired_design_fields_match"] = {
        "pass": len(pair_field_mismatches) == 0,
        "violations": pair_field_mismatches,
    }

    # 4) no standalone را in context
    ra_violations = [
        {"item_id": r["item_id"], "condition": r["condition"], "context": r["context"]}
        for r in rows
        if has_standalone_ra(r["context"])
    ]
    summary["checks"]["no_standalone_ra_in_context"] = {
        "pass": len(ra_violations) == 0,
        "violations": ra_violations,
    }

    # 5) object appears in context
    object_missing = [
        {"item_id": r["item_id"], "condition": r["condition"], "object": r["object"], "context": r["context"]}
        for r in rows
        if normalize_space(r["object"]) not in normalize_space(r["context"])
    ]
    summary["checks"]["object_appears_in_context"] = {
        "pass": len(object_missing) == 0,
        "violations": object_missing,
    }

    # 6) verb appears in context
    verb_missing = [
        {"item_id": r["item_id"], "condition": r["condition"], "verb": r["verb"], "context": r["context"]}
        for r in rows
        if normalize_space(r["verb"]) not in normalize_space(r["context"])
    ]
    summary["checks"]["verb_appears_in_context"] = {
        "pass": len(verb_missing) == 0,
        "violations": verb_missing,
    }

    # 7) anaphor type set
    allowed = {"same_np", "demonstrative_np"}
    anaphor_bad = [
        {"item_id": r["item_id"], "condition": r["condition"], "object_anaphor_type": r["object_anaphor_type"]}
        for r in rows
        if r["object_anaphor_type"] not in allowed
    ]
    summary["checks"]["allowed_object_anaphor_types_only"] = {
        "pass": len(anaphor_bad) == 0,
        "violations": anaphor_bad,
    }

    # 8/9) marker-only target span and full-NP continuation prefix constraints
    same_bad = []
    dem_bad = []
    for r in rows:
        if r["object_anaphor_type"] == "same_np":
            expected_marker = "همان"
            expected_prefix = f"همان {r['object']}"
        else:
            expected_marker = "این"
            expected_prefix = f"این {r['object']}"

        target_ok = normalize_space(r["target_span"]) == normalize_space(expected_marker)
        cont_ok = normalize_space(r["continuation_object_anaphoric"]).startswith(
            normalize_space(expected_prefix)
        )

        if r["object_anaphor_type"] == "same_np" and (not target_ok or not cont_ok):
            same_bad.append(
                {
                    "item_id": r["item_id"],
                    "condition": r["condition"],
                    "target_span": r["target_span"],
                    "expected_marker_target_span": expected_marker,
                    "expected_full_np_prefix": expected_prefix,
                    "continuation_object_anaphoric": r["continuation_object_anaphoric"],
                }
            )
        if r["object_anaphor_type"] == "demonstrative_np" and (not target_ok or not cont_ok):
            dem_bad.append(
                {
                    "item_id": r["item_id"],
                    "condition": r["condition"],
                    "target_span": r["target_span"],
                    "expected_marker_target_span": expected_marker,
                    "expected_full_np_prefix": expected_prefix,
                    "continuation_object_anaphoric": r["continuation_object_anaphoric"],
                }
            )

    summary["checks"]["same_np_marker_target_and_full_np_prefix"] = {
        "pass": len(same_bad) == 0,
        "violations": same_bad,
    }
    summary["checks"]["demonstrative_np_marker_target_and_full_np_prefix"] = {
        "pass": len(dem_bad) == 0,
        "violations": dem_bad,
    }

    # 6b) subject name must not appear in object continuation
    subject_leaks = [
        {
            "item_id": r["item_id"],
            "condition": r["condition"],
            "subject": r["subject"],
            "continuation_object_anaphoric": r["continuation_object_anaphoric"],
        }
        for r in rows
        if normalize_space(r["subject"]) in normalize_space(r["continuation_object_anaphoric"])
    ]
    summary["checks"]["no_subject_name_in_object_continuation"] = {
        "pass": len(subject_leaks) == 0,
        "violations": subject_leaks,
    }

    # 10) each verb has 4 items total, 2 per anaphor type (item-level via bare rows)
    bare_rows = [r for r in rows if r["condition"] == "bare"]
    verb_item_counts = Counter(r["verb"] for r in bare_rows)
    verb_type_counts: Dict[str, Counter] = defaultdict(Counter)
    for r in bare_rows:
        verb_type_counts[r["verb"]][r["object_anaphor_type"]] += 1

    verb_balance_bad = []
    for verb in sorted(verb_item_counts):
        total = verb_item_counts[verb]
        same_n = verb_type_counts[verb]["same_np"]
        dem_n = verb_type_counts[verb]["demonstrative_np"]
        if not (total == 4 and same_n == 2 and dem_n == 2):
            verb_balance_bad.append(
                {
                    "verb": verb,
                    "total_items": total,
                    "same_np_items": same_n,
                    "demonstrative_np_items": dem_n,
                }
            )

    summary["checks"]["verb_balance_4_total_2_2_split"] = {
        "pass": len(verb_balance_bad) == 0,
        "violations": verb_balance_bad,
    }

    # 11/12) no exact repetition of subject/object continuations at item-level (bare rows)
    sc_counts = Counter(normalize_space(r["continuation_subject_continuing"]) for r in bare_rows)
    oc_counts = Counter(normalize_space(r["continuation_object_anaphoric"]) for r in bare_rows)

    sc_repeats = [{"text": k, "count": v} for k, v in sc_counts.items() if v > 1]
    oc_repeats = [{"text": k, "count": v} for k, v in oc_counts.items() if v > 1]

    summary["checks"]["no_exact_subject_continuation_repeats"] = {
        "pass": len(sc_repeats) == 0,
        "violations": sc_repeats,
    }
    summary["checks"]["no_exact_object_continuation_repeats"] = {
        "pass": len(oc_repeats) == 0,
        "violations": oc_repeats,
    }

    # 12b) heuristic subject/object continuation content-word overlap review
    continuation_overlap_candidates = []
    for r in bare_rows:
        subj_words = set(content_words(r["continuation_subject_continuing"]))
        obj_words = set(content_words(r["continuation_object_anaphoric"]))
        shared = sorted(subj_words & obj_words)
        if shared:
            continuation_overlap_candidates.append(
                {
                    "item_id": r["item_id"],
                    "shared_content_words": shared,
                    "continuation_subject_continuing": r["continuation_subject_continuing"],
                    "continuation_object_anaphoric": r["continuation_object_anaphoric"],
                }
            )

    object_domain_counts = Counter(r["object_domain"] for r in bare_rows)
    object_domain_by_anaphor: Dict[str, Counter] = defaultdict(Counter)
    for r in bare_rows:
        object_domain_by_anaphor[r["object_domain"]][r["object_anaphor_type"]] += 1

    empty_domain_rows = [
        {"item_id": r["item_id"], "condition": r["condition"]}
        for r in rows
        if not normalize_space(r["object_domain"])
    ]
    summary["checks"]["object_domain_nonempty"] = {
        "pass": len(empty_domain_rows) == 0,
        "violations": empty_domain_rows,
    }

    summary["review_checks"]["subject_object_continuation_content_overlap"] = {
        "candidate_count": len(continuation_overlap_candidates),
        "candidates": continuation_overlap_candidates,
    }
    summary["review_checks"]["object_domain_distribution_bare_only"] = {
        "counts": dict(object_domain_counts),
        "by_anaphor_type": {
            domain: dict(object_domain_by_anaphor[domain])
            for domain in sorted(object_domain_by_anaphor)
        },
    }

    # Counts block
    summary["counts"] = {
        "rows_by_condition": Counter(r["condition"] for r in rows),
        "rows_by_anaphor_type": Counter(r["object_anaphor_type"] for r in rows),
        "item_verbs_bare_only": verb_item_counts,
        "item_object_domains_bare_only": Counter(r["object_domain"] for r in bare_rows),
        "item_verb_anaphor_split_bare_only": {
            v: dict(verb_type_counts[v]) for v in sorted(verb_type_counts)
        },
    }

    # Convert Counters to dict for JSON serialization.
    summary["counts"]["rows_by_condition"] = dict(summary["counts"]["rows_by_condition"])
    summary["counts"]["rows_by_anaphor_type"] = dict(summary["counts"]["rows_by_anaphor_type"])
    summary["counts"]["item_verbs_bare_only"] = dict(summary["counts"]["item_verbs_bare_only"])
    summary["counts"]["item_object_domains_bare_only"] = dict(summary["counts"]["item_object_domains_bare_only"])

    all_pass = all(v["pass"] for v in summary["checks"].values())
    summary["all_required_checks_pass"] = all_pass

    if not all_pass:
        issues.append("One or more required checks failed. See checks section.")

    return summary, issues


def main() -> None:
    print("Starting continuation_yek v5.5.3 validation")
    print(f"Repository root: {ROOT}")
    print(f"Input CSV: {INPUT_CSV}")
    print(f"Output JSON: {OUT_JSON}")

    ensure_v553_csv_exists()
    rows = load_rows(INPUT_CSV)
    summary, issues = run_checks(rows)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")


    print(f"Wrote: {OUT_JSON}")
    print(f"all_required_checks_pass={summary['all_required_checks_pass']}")
    print(f"validated_at_utc={summary['validated_at_utc']}")


if __name__ == "__main__":
    main()
