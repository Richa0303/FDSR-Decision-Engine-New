# eval_qa.py
# Offline Q&A evaluation against your data.json fields.
# Scores (per QA type and overall):
# - method_hit@1        : top-1 retrieved method == seed method
# - section_nonempty@1  : top-1 method has a non-empty target field
# - token_F1            : token-level F1 between predicted text and gold text
# - ROUGE-L             : LCS-based recall between predicted and gold
#
# TSV expected columns:
#   qid, query, type, seed
# Optional: field  (if present, overrides mapping below)
#
# Example types -> fields mapping:
#   QA_DEF        -> long_description (fallback short_description)
#   QA_STEPS      -> steps
#   QA_TIME_RES   -> time_cost + resource_cost
#   QA_OUTPUTS    -> outputs (+ deliverables)
#   QA_RISKS      -> risks_biases
#   QA_PREREQ     -> prerequisites
#   QA_WHEN       -> when_to_use
#   QA_PARTICIPANTS-> participation
#   QA_KEYWORDS   -> keywords
#   QA_PHASE      -> phase (or phase_membership summary)
#   QA_SOURCE     -> source
#
# Usage:
#   python eval_qa.py --tsv qa.tsv

import argparse, csv, json, re, statistics
from typing import Dict, List, Tuple, Optional
from search_engine import build_or_load_index, search_methods

# ---------- Helpers ----------
def _norm(s): return (s or "").strip()
def _lc(s): return (s or "").lower().strip()

TYPE2FIELD = {
    "QA_DEF": "long_description",
    "QA_STEPS": "steps",
    "QA_TIME_RES": "time_resource",
    "QA_OUTPUTS": "outputs",
    "QA_RISKS": "risks_biases",
    "QA_PREREQ": "prerequisites",
    "QA_WHEN": "when_to_use",
    "QA_PARTICIPANTS": "participation",
    "QA_KEYWORDS": "keywords",
    "QA_PHASE": "phase",
    "QA_SOURCE": "source",
}

def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())

def _token_f1(pred: str, gold: str) -> float:
    p = _tokens(pred); g = _tokens(gold)
    if not p and not g: return 1.0
    if not p or not g:  return 0.0
    pset, gset = set(p), set(g)
    inter = len(pset & gset)
    prec = inter / max(1, len(pset))
    rec  = inter / max(1, len(gset))
    return (2*prec*rec / max(prec+rec, 1e-12))

def _lcs(a: List[str], b: List[str]) -> int:
    # classic DP LCS length
    n, m = len(a), len(b)
    dp = [0]*(m+1)
    for i in range(1, n+1):
        prev = 0
        for j in range(1, m+1):
            temp = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j-1])
            prev = temp
    return dp[m]

def _rougeL_recall(pred: str, gold: str) -> float:
    # ROUGE-L recall = LCS(gold, pred) / len(gold)
    g = _tokens(gold); p = _tokens(pred)
    if not g: return 1.0 if not p else 0.0
    return _lcs(g, p) / len(g)

def _first_nonempty(*vals) -> Optional[str]:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, list) and len([x for x in v if str(x).strip()]) > 0:
            return "; ".join(str(x) for x in v if str(x).strip())
    return None

def _field_text(method: dict, field_key: str) -> str:
    """Render method[field] into a plain string for scoring."""

    # 1) Steps fallback: steps -> key_steps
    if field_key == "steps":
        v = method.get("steps") or method.get("key_steps")
        if isinstance(v, list):
            return "; ".join(str(x) for x in v if str(x).strip())
        return v or ""

    # 2) Time + Resources combined view used by QA_TIME_RES
    if field_key == "time_resource":
        t = method.get("time_cost")
        r = method.get("resource_cost")
        if t or r:
            return f"Time: {t}; Resources: {r}".strip("; ")
        return ""

    # 3) Outputs should also consider deliverables
    if field_key == "outputs":
        v = method.get("outputs")
        if not v:
            v = method.get("deliverables")
        if isinstance(v, list):
            return "; ".join(str(x) for x in v if str(x).strip())
        return v or ""

    # 4) Phase: return a single dominant phase if possible; else membership summary
    if field_key == "phase":
        phase = method.get("phase")
        if phase:
            return phase
        pm = method.get("phase_membership", {}) or {}
        if pm:
            # dominant phase name
            return max(pm.items(), key=lambda kv: float(kv[1]))[0]
        return ""

    # ---- generic fallbacks ----
    v = method.get(field_key)
    if isinstance(v, list):
        return "; ".join(str(x) for x in v if str(x).strip())
    if isinstance(v, str):
        return v
    return str(v) if v is not None else ""


def load_data(path="data.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def index_by_title(data: List[dict]) -> Dict[str, dict]:
    return { _lc(m.get("title","")): m for m in data if m.get("title") }

def read_qa_tsv(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            rows.append({
                "qid": _norm(r.get("qid")),
                "query": _norm(r.get("query")),
                "type": _norm(r.get("type") or r.get("intent")),
                "seed": _norm(r.get("seed")),
                "field": _norm(r.get("field")),   # optional
            })
    return rows

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, help="QA TSV with columns: qid,query,type,seed[,field]")
    ap.add_argument("--k", type=int, default=1, help="use top-k? (we score top-1 method_hit; text from top-1)")
    args = ap.parse_args()

    INDEX, META = build_or_load_index()
    DATA = load_data()
    by_title = index_by_title(DATA)

    bench = read_qa_tsv(args.tsv)

    # aggregators per type
    per_type = {}
    def ensure(t):
        if t not in per_type:
            per_type[t] = {
                "count": 0,
                "method_hit@1": [],
                "section_nonempty@1": [],
                "token_F1": [],
                "rougeL": [],
            }

    overall = {"count": 0, "method_hit@1": [], "section_nonempty@1": [], "token_F1": [], "rougeL": []}

    for row in bench:
        q = row["query"]; t = row["type"]; seed = row["seed"]; fld = row["field"]
        ensure(t)

        # determine target field
        field_key = fld or TYPE2FIELD.get(t)
        if not field_key:
            # unknown type -> skip
            continue

        gold_method = by_title.get(_lc(seed))
        gold_text = ""
        if gold_method:
            # gold: render field from the *seed* method
            if field_key == "QA_DEF":  # never happens; just guard
                field_key = "long_description"
            if field_key == "long_description":
                gold_text = _first_nonempty(gold_method.get("long_description"), gold_method.get("short_description")) or ""
            else:
                gold_text = _field_text(gold_method, field_key)

        # predict via your app brain
        out = search_methods(q, INDEX, META, k=max(5, args.k))
        results = out[0] if isinstance(out, tuple) else out
        top1 = results[0] if results else {}
        pred_title = top1.get("title","")
        pred_text = ""
        if top1:
            if field_key == "long_description":
                pred_text = _first_nonempty(top1.get("long_description"), top1.get("short_description")) or ""
            else:
                pred_text = _field_text(top1, field_key)

        method_hit = 1.0 if _lc(pred_title) == _lc(seed) else 0.0
        section_nonempty = 1.0 if (pred_text.strip()) else 0.0

        f1 = _token_f1(pred_text, gold_text) if gold_text else 0.0
        rL = _rougeL_recall(pred_text, gold_text) if gold_text else 0.0

        # record
        per_type[t]["count"] += 1
        per_type[t]["method_hit@1"].append(method_hit)
        per_type[t]["section_nonempty@1"].append(section_nonempty)
        per_type[t]["token_F1"].append(f1)
        per_type[t]["rougeL"].append(rL)

        overall["count"] += 1
        overall["method_hit@1"].append(method_hit)
        overall["section_nonempty@1"].append(section_nonempty)
        overall["token_F1"].append(f1)
        overall["rougeL"].append(rL)

    def _summ(block):
        if block["count"] == 0:
            return {"count": 0, "note": "No QA rows"}
        return {
            "count": block["count"],
            "method_hit@1": round(statistics.mean(block["method_hit@1"]), 6),
            "section_nonempty@1": round(statistics.mean(block["section_nonempty@1"]), 6),
            "token_F1_mean": round(statistics.mean(block["token_F1"]), 6),
            "rougeL_mean": round(statistics.mean(block["rougeL"]), 6),
        }

    result = {"OVERALL": _summ(overall)}
    for t, blk in per_type.items():
        result[t] = _summ(blk)

    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
