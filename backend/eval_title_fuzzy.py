# eval.py
# -*- coding: utf-8 -*-
"""
Evaluate TITLE + FUZZY benchmarks.

TSV format (tab-separated):
qid    query    type    seed    k    gold

Types:
- TITLE: exact title lookup; 'gold' should contain the canonical method title
         (if gold is empty, fall back to 'seed').
- FUZZY: lenient title lookup; 'seed' holds the intended canonical title,
         'gold' is usually '#'.

Metrics reported (per type and overall):
- EM@1   : exact match at rank 1 (normalized title comparison)
- Hits@k : 1 if target appears in top-k, else 0
- MRR@k  : reciprocal rank if found within top-k, else 0
"""

import csv
import argparse
import statistics
import re
from typing import List, Dict, Tuple, Optional

from search_engine import build_or_load_index, search_methods

# -------------------------
# Normalization helpers
# -------------------------
_ws_re = re.compile(r"\s+")
_punct_re = re.compile(r"[^\w\s]")

def norm_title(s: str) -> str:
    """Lowercase, strip punctuation, collapse spaces."""
    if not s:
        return ""
    s = s.lower()
    s = _punct_re.sub(" ", s)
    s = _ws_re.sub(" ", s).strip()
    return s

# -------------------------
# Metrics helpers
# -------------------------
def em1(pred_top1: str, gold: str) -> float:
    return 1.0 if norm_title(pred_top1) == norm_title(gold) else 0.0

def hits_at_k(preds: List[str], gold: str, k: int) -> float:
    gold_n = norm_title(gold)
    for p in preds[:k]:
        if norm_title(p) == gold_n:
            return 1.0
    return 0.0

def mrr_at_k(preds: List[str], gold: str, k: int) -> float:
    gold_n = norm_title(gold)
    for i, p in enumerate(preds[:k], start=1):
        if norm_title(p) == gold_n:
            return 1.0 / i
    return 0.0

# -------------------------
# Evaluation
# -------------------------
def evaluate_rows(rows: List[Dict], k_default: int = 5) -> Dict:
    index, meta = build_or_load_index()

    per_type = {}  # type -> list of metric dicts
    all_metrics = []

    for row in rows:
        qid   = row.get("qid", "")
        query = row.get("query", "")
        qtype = (row.get("type", "") or "").strip().upper()
        seed  = row.get("seed", "") or ""
        gold  = row.get("gold", "") or ""
        k_str = row.get("k", "").strip()
        k = int(k_str) if k_str.isdigit() else k_default

        # Determine gold target per type
        if qtype == "TITLE":
            target = gold or seed
        elif qtype == "FUZZY":
            # For FUZZY, use 'seed' as the canonical intended title
            target = seed
        else:
            # Unknown types: skip gracefully
            # (If you add more types later, extend this block)
            continue

        # Perform search
        results, _constraints = search_methods(query, index, meta, k=max(k, 10))
        pred_titles = [r.get("title", "") for r in results]
        top1 = pred_titles[0] if pred_titles else ""

        # Compute metrics
        m = {
            "qid": qid,
            "type": qtype,
            "em1": em1(top1, target) if pred_titles else 0.0,
            "hits@k": hits_at_k(pred_titles, target, k) if pred_titles else 0.0,
            "mrr@k": mrr_at_k(pred_titles, target, k) if pred_titles else 0.0,
        }
        all_metrics.append(m)
        per_type.setdefault(qtype, []).append(m)

    def _agg(ms: List[Dict]) -> Dict:
        if not ms:
            return {"count": 0, "em1": 0.0, "hits@k": 0.0, "mrr@k": 0.0}
        return {
            "count": len(ms),
            "em1": statistics.mean(m["em1"] for m in ms),
            "hits@k": statistics.mean(m["hits@k"] for m in ms),
            "mrr@k": statistics.mean(m["mrr@k"] for m in ms),
        }

    summary = {t: _agg(ms) for t, ms in per_type.items()}
    summary["OVERALL"] = _agg(all_metrics)
    return summary

# -------------------------
# CLI
# -------------------------
def read_tsv(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(r)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, help="Path to benchmark TSV (TITLE + FUZZY).")
    ap.add_argument("--k", type=int, default=5, help="Default k if not provided per row.")
    args = ap.parse_args()

    rows = read_tsv(args.tsv)
    summary = evaluate_rows(rows, k_default=args.k)

    # Pretty print
    print("{")
    first = True
    for key, val in summary.items():
        if not first: print(",")
        first = False
        print(f'  "{key}": ' + "{"
              f'"count": {val["count"]}, '
              f'"EM@1": {val["em1"]:.4f}, '
              f'"Hits@k": {val["hits@k"]:.4f}, '
              f'"MRR@k": {val["mrr@k"]:.4f}'
              "}", end="")
    print("\n}")

if __name__ == "__main__":
    main()
