# eval_similarity.py
# Evaluate similarity queries (your TSV schema) with standard IR metrics.
import argparse, csv, math, statistics, json
from search_engine import build_or_load_index, search_methods

def norm(s): return (s or "").strip()

def dcg(rels):
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))

def ndcg_at_k(pred, gold, k):
    rels = [1 if t in gold else 0 for t in pred[:k]]
    ideal = sorted(rels, reverse=True)
    idcg = dcg(ideal)
    return (dcg(rels) / idcg) if idcg > 0 else 0.0

def ap_at_k(pred, gold, k):
    hits, precs = 0, []
    for i, t in enumerate(pred[:k], 1):
        if t in gold:
            hits += 1
            precs.append(hits / i)
    # divide by |gold| to be conservative when multiple gold labels exist
    return (sum(precs) / max(1, len(gold))) if gold else 0.0

def recall_at_k(pred, gold, k):
    return len(set(pred[:k]) & gold) / max(1, len(gold))

def load_titles_map(data_json_path="data.json"):
    with open(data_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    canon = {norm(m["title"]): m["title"] for m in data if m.get("title")}
    lower = {k.lower(): v for k, v in canon.items()}
    return canon, lower

def canonicalize(title, lower_map):
    t = norm(title)
    return lower_map.get(t.lower(), t)  # leave as-is if unseen (validator should catch)

def read_benchmark(path, lower_map):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            q = norm(r["query"])
            k = int(r["topk"])
            # canonicalize all answer titles
            gold = {canonicalize(a, lower_map) for a in r["answers"].split(";") if norm(a)}
            rows.append((q, k, gold))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, help="TSV with columns: qid,query,intent,seed,topk,answers")
    ap.add_argument("--k", type=int, default=None, help="Override K (else per-row topk is used)")
    args = ap.parse_args()

    (INDEX, META) = build_or_load_index()
    _, lower = load_titles_map()

    bench = read_benchmark(args.tsv, lower)
    recalls, maps, ndcgs = [], [], []
    unique_recs = set()
    intralist_sims = []

    def toks(t): return {w for w in t.lower().split() if w.isalnum()}

    for q, row_k, gold in bench:
        K = args.k or row_k
        res = search_methods(q, INDEX, META)
        if isinstance(res, tuple) and len(res) == 2:
            res = res[0]
        preds = [canonicalize(r.get("title",""), lower) for r in res][:K]

        recalls.append(recall_at_k(preds, gold, K))
        maps.append(ap_at_k(preds, gold, K))
        ndcgs.append(ndcg_at_k(preds, gold, K))

        for t in preds: 
            if t: unique_recs.add(t)

        # crude intra-list similarity (token Jaccard)
        S = [toks(t) for t in preds if t]
        pairs = 0; sim = 0.0
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                a, b = S[i], S[j]
                if a and b:
                    sim += len(a & b) / len(a | b)
                    pairs += 1
        intralist_sims.append(sim / pairs if pairs else 0.0)

    out = {
        "num_queries": len(bench),
        "recall@k_mean": statistics.mean(recalls),
        "map@k_mean": statistics.mean(maps),
        "ndcg@k_mean": statistics.mean(ndcgs),
        "coverage_unique": len(unique_recs),
        "coverage_ratio": len(unique_recs) / max(1, len(META)),
        "intralist_similarity_mean": statistics.mean(intralist_sims),
    }
    print(out)

if __name__ == "__main__":
    main()
