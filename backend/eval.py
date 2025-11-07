# -*- coding: utf-8 -*-
"""
Similarity (SML) evaluation for "similar to ..." queries.

Input TSV columns (tab-separated):
    qid    query    intent    seed    topk    answers

- query: a natural-language query (e.g., "similar to Delphi Method")
- topk : integer K to evaluate at (defaults to 10 if blank)
- answers: semicolon-separated gold titles (DO NOT include the seed)

Metrics reported:
- recall@K           : |pred ∩ gold| / |gold|
- map@K              : mean average precision at K (binary relevance)
- ndcg@K             : normalized DCG at K (binary relevance)
- coverage_unique    : # unique items the system retrieved across all queries
- coverage_ratio     : coverage_unique / total_catalog_size
- intralist_similarity_mean : average pairwise title-token Jaccard in top-K (lower is more diverse)
"""

import csv, json, math, statistics
from typing import List, Set, Dict
from search_engine import build_or_load_index, search_methods

def _split_titles(s: str) -> List[str]:
    return [t.strip() for t in (s or "").split(";") if t.strip()]

def _jaccard(a: str, b: str) -> float:
    ta = set([t for t in a.lower().replace("–","-").split() if t])
    tb = set([t for t in b.lower().replace("–","-").split() if t])
    if not ta or not tb: return 0.0
    return len(ta & tb) / len(ta | tb)

def _dcg_at_k(rels: List[int]) -> float:
    dcg = 0.0
    for i, r in enumerate(rels, start=1):
        if r > 0:
            dcg += (2**r - 1) / math.log2(i + 1)
    return dcg

def _ndcg_at_k(binary_rels: List[int]) -> float:
    dcg = _dcg_at_k(binary_rels)
    ideal = _dcg_at_k(sorted(binary_rels, reverse=True))
    return dcg / ideal if ideal > 0 else 0.0

def _ap_at_k(binary_rels: List[int]) -> float:
    # Average Precision at K with binary relevance
    num_rel = 0
    precisions = []
    for i, r in enumerate(binary_rels, start=1):
        if r:
            num_rel += 1
            precisions.append(num_rel / i)
    return sum(precisions) / max(num_rel, 1) if precisions else 0.0

def main(tsv_path: str, k_default: int = 10,
         use_mmr: bool = False, mmr_lambda: float = 0.75,
         similar_phase_bias: float = 0.15):
    INDEX, META = build_or_load_index()
    catalog_titles = { (m.get("title") or "").strip() for m in META }

    rows = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            if (r.get("intent") or "").strip().upper() != "SML":
                continue
            rows.append(r)

    recall_list, map_list, ndcg_list = [], [], []
    all_retrieved_titles: Set[str] = set()
    ils_vals = []

    for r in rows:
        q = (r.get("query") or "").strip()
        gold = set(_split_titles(r.get("answers") or ""))
        K = int(r.get("topk") or k_default)

        # Run retrieval in cosine mode (pure similarity), with seed-aware bias
        results, _ = search_methods(
            q, INDEX, META, k=K, fetch_k=max(100, 10*K),
            mode="cosine",
            use_mmr=use_mmr, mmr_lambda=mmr_lambda,
            similar_phase_bias=similar_phase_bias
        )

        pred_titles = [(m.get("title") or "").strip() for m in results[:K]]
        all_retrieved_titles.update(pred_titles)

        # Binary relevance vs. gold
        binary = [1 if t in gold else 0 for t in pred_titles]

        # Metrics
        hit_rel = sum(binary)
        recall = hit_rel / max(len(gold), 1)
        ap = _ap_at_k(binary)
        ndcg = _ndcg_at_k(binary)

        recall_list.append(recall)
        map_list.append(ap)
        ndcg_list.append(ndcg)

        # Intra-list similarity (title-token Jaccard)
        if len(pred_titles) >= 2:
            pairs = 0
            acc = 0.0
            for i in range(len(pred_titles)):
                for j in range(i+1, len(pred_titles)):
                    acc += _jaccard(pred_titles[i], pred_titles[j])
                    pairs += 1
            ils_vals.append(acc / max(pairs, 1))
        else:
            ils_vals.append(0.0)

    out = {
        "SML": {
            "count": len(rows),
            "recall@k_mean": round(statistics.mean(recall_list), 6) if rows else 0.0,
            "map@k_mean": round(statistics.mean(map_list), 6) if rows else 0.0,
            "ndcg@k_mean": round(statistics.mean(ndcg_list), 6) if rows else 0.0,
            "intralist_similarity_mean": round(statistics.mean(ils_vals), 6) if rows else 0.0,
            "coverage_unique": len(all_retrieved_titles),
            "coverage_ratio": round(len(all_retrieved_titles) / max(len(catalog_titles),1), 6),
        }
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, help="SML TSV with columns: qid query intent seed topk answers")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--use_mmr", action="store_true")
    ap.add_argument("--mmr_lambda", type=float, default=0.75)
    ap.add_argument("--similar_phase_bias", type=float, default=0.15)
    args = ap.parse_args()
    main(args.tsv, k_default=args.k, use_mmr=args.use_mmr,
         mmr_lambda=args.mmr_lambda, similar_phase_bias=args.similar_phase_bias)
