# beir_sanity.py
# Minimal BEIR retrieval eval using SentenceTransformers + FAISS.
# Reads: --corpus corpus.jsonl[.gz], --queries queries.jsonl[.gz], --qrels qrels/test.tsv

import argparse, json, csv, math, statistics, os, gzip, io
from typing import Dict, List, Tuple, Set
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def _open_text(path: str):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")

def load_corpus(path:str) -> Tuple[List[str], List[str]]:
    """Returns (doc_ids, texts). BEIR JSONL has fields: _id, title, text."""
    doc_ids, texts = [], []
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            did = str(obj.get("_id"))
            title = obj.get("title") or obj.get("name") or ""
            body  = obj.get("text")  or obj.get("contents") or ""
            text = (f"{title} {body}").strip()
            doc_ids.append(did); texts.append(text)
    return doc_ids, texts

def load_queries(path:str) -> Dict[str, str]:
    q = {}
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj.get("_id"))
            qtext = obj.get("text") or obj.get("query") or ""
            q[qid] = qtext
    return q

def load_qrels(path:str) -> Dict[str, Set[str]]:
    """TSV with columns: qid  docid  relevance (>=1 means relevant)."""
    rels: Dict[str, Set[str]] = {}
    with _open_text(path) as f:
        rdr = csv.reader(f, delimiter="\t")
        first = True
        for row in rdr:
            if not row:
                continue
            if first and row[0].lower() in {"qid", "query-id"}:
                first = False
                continue
            first = False
            qid   = str(row[0])
            docid = str(row[1])
            score = int(row[2]) if len(row) > 2 and row[2].strip().isdigit() else 1
            if score >= 1:
                rels.setdefault(qid, set()).add(docid)
    return rels

def dcg(rels: List[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))

def ndcg_at_k(pred: List[str], gold: Set[str], k: int) -> float:
    rels = [1 if d in gold else 0 for d in pred[:k]]
    ideal = sorted(rels, reverse=True)
    idcg = dcg(ideal)
    return (dcg(rels) / idcg) if idcg > 0 else 0.0

def ap_at_k(pred: List[str], gold: Set[str], k: int) -> float:
    hits, precs = 0, []
    for i, d in enumerate(pred[:k], 1):
        if d in gold:
            hits += 1
            precs.append(hits / i)
    return (sum(precs) / max(1, len(gold))) if gold else 0.0

def recall_at_k(pred: List[str], gold: Set[str], k: int) -> float:
    return len(set(pred[:k]) & gold) / max(1, len(gold))

def _apply_bge_prompt(texts: List[str], is_query: bool) -> List[str]:
    if is_query:
        prefix = "Represent this sentence for searching relevant passages: "
    else:
        prefix = "Represent this passage for retrieval: "
    return [prefix + (t or "") for t in texts]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--model", default="all-MiniLM-L6-v2",
                    help="Any SentenceTransformers model, e.g., 'BAAI/bge-small-en-v1.5'")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--normalize", action="store_true",
                    help="L2-normalize before indexing. (If not set, we'll normalize anyway for cosine IP.)")
    ap.add_argument("--use-bge-prompts", action="store_true",
                    help="Apply BGE query/passage prompts (recommended for BGE models).")
    args = ap.parse_args()

    # 1) Load BEIR data
    doc_ids, texts = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    qrels   = load_qrels(args.qrels)

    dataset_name = os.path.basename(os.path.dirname(os.path.abspath(args.corpus))) or "beir-set"
    print(f"Loaded: corpus={len(doc_ids)} docs, queries={len(queries)}, qrels(qids)={len(qrels)}")

    # 2) Embed & index corpus with your model
    model = SentenceTransformer(args.model)

    corpus_texts = _apply_bge_prompt(texts, is_query=False) if args.use_bge_prompts else texts
    emb = model.encode(
        corpus_texts,
        batch_size=args.batch,
        normalize_embeddings=args.normalize,
        show_progress_bar=True,
    )
    emb = np.asarray(emb, dtype="float32")
    if not args.normalize:
        # normalize for cosine-like inner product
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        emb = emb / norms

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    # 3) Evaluate
    R, MAP, NDCG = [], [], []
    evaluated = 0
    for qid, qtext in queries.items():
        gold = qrels.get(qid)
        if not gold:
            continue

        qtxt = qtext
        if args.use_bge_prompts:
            qtxt = "Represent this sentence for searching relevant passages: " + (qtext or "")

        qvec = model.encode([qtxt], normalize_embeddings=True).astype("float32")
        scores, idxs = index.search(qvec, args.k)
        preds = [doc_ids[i] for i in idxs[0]]

        R.append(recall_at_k(preds, gold, args.k))
        MAP.append(ap_at_k(preds, gold, args.k))
        NDCG.append(ndcg_at_k(preds, gold, args.k))
        evaluated += 1

    result = {
        "dataset": dataset_name,
        "model": args.model,
        "k": args.k,
        "num_queries_scored": evaluated,
        "recall@k_mean": round(statistics.mean(R), 6) if R else 0.0,
        "map@k_mean": round(statistics.mean(MAP), 6) if MAP else 0.0,
        "ndcg@k_mean": round(statistics.mean(NDCG), 6) if NDCG else 0.0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
