# search_engine.py
# -*- coding: utf-8 -*-

import json
import os
import re
import math
import hashlib
import difflib
import string
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# =========================
# Globals & paths
# =========================

MODEL_NAME = "BAAI/bge-small-en-v1.5"
# Re-ranker model (pairwise query, doc)
CE_MODEL_NAME = os.getenv("CE_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
_ce_model = None
_ce_tok = None
def _doc_text(m: Dict) -> str:
    """Richer document text for indexing & similarity."""
    parts = [
        m.get("title",""),
        m.get("short_description",""),
        m.get("long_description",""),
        " ".join(m.get("keywords",[]) or []),
        " ".join(m.get("when_to_use",[]) or []),
        " ".join(m.get("thought_styles",[]) or []),
        " ".join(m.get("participation",[]) or []),
    ]
    st = m.get("steps")
    if isinstance(st, list): parts.append(" ".join(st))
    elif isinstance(st, str): parts.append(st)
    return " ".join(p for p in parts if p).strip()

def _dominant_phase(pm: Dict[str, float]) -> Optional[str]:
    if not pm:
        return None
    # phases we use everywhere else
    phases = ["problem_framing", "implementation", "evaluation", "conclusion"]
    top_p, top_v = None, -1.0
    for p in phases:
        v = float(pm.get(p, 0.0))
        if v > top_v:
            top_p, top_v = p, v
    return top_p

# --- lightweight diversity for top-k (MMR over bag-of-words) ---
def _cosine_like(a: dict, b: dict) -> float:
    import re
    def toks(m):
        txt = " ".join([
            m.get("title",""), m.get("short_description",""), m.get("long_description",""),
            " ".join(m.get("keywords",[]) or []),
            " ".join(m.get("when_to_use",[]) or []),
            " ".join(m.get("thought_styles",[]) or []),
            " ".join(m.get("participation",[]) or []),
        ]).lower()
        return set(re.findall(r"[a-z0-9]{3,}", txt))
    A, B = toks(a), toks(b)
    if not A or not B: 
        return 0.0
    inter = len(A & B); uni = len(A | B)
    return inter / max(1, uni)

def mmr_diversify(items: List[dict], k: int = 8, lambda_: float = 0.75) -> List[dict]:
    if not items:
        return []
    # seed with the highest score
    selected = [items[0]]
    cand = items[1:]
    while cand and len(selected) < k:
        best_i, best_val = 0, -1e9
        for i, it in enumerate(cand):
            rel = float(it.get("score", 0.0))
            div = max(_cosine_like(it, s) for s in selected) if selected else 0.0
            val = lambda_ * rel - (1.0 - lambda_) * div
            if val > best_val:
                best_val, best_i = val, i
        selected.append(cand.pop(best_i))
    return selected

def _get_ce():
    global _ce_model, _ce_tok
    if _ce_model is None:
        _ce_tok = AutoTokenizer.from_pretrained(CE_MODEL_NAME)
        _ce_model = AutoModelForSequenceClassification.from_pretrained(CE_MODEL_NAME)
        _ce_model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    return _ce_model, _ce_tok
def _rerank_cross_encoder(query: str, docs: List[dict], text_fn=_doc_text, batch_size: int = 32) -> List[float]:
    model, tok = _get_ce()
    dev = next(model.parameters()).device
    pairs_q = [query] * len(docs)
    pairs_d = [text_fn(m) for m in docs]

    scores: List[float] = []
    for i in range(0, len(docs), batch_size):
        q_batch = pairs_q[i:i+batch_size]
        d_batch = pairs_d[i:i+batch_size]
        inputs = tok(q_batch, d_batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
        with torch.no_grad():
            logits = model(**inputs).logits.squeeze(-1).float().cpu().numpy().tolist()
        if isinstance(logits, float): logits = [logits]
        scores.extend(logits)
    return scores  # higher is better

DATA_PATH = os.path.join(os.path.dirname(__file__), "data.json")
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "faiss_index.bin")
META_PATH = os.path.join(os.path.dirname(__file__), "faiss_meta.json")
HASH_PATH = os.path.join(os.path.dirname(__file__), "data.hash")

# Load the embedding model once
_model = SentenceTransformer(MODEL_NAME)

# =========================
# Query intents & utilities
# =========================

_SIMILAR_RE = re.compile(r'\b(similar to|methods like|related to|alternatives to)\s+(.+)$', re.I)
_DASH = r"[-–—]"  # hyphen, en dash, em dash

def _clean_name(s: str) -> str:
    """Strip quotes/punctuation/whitespace from a seed method name."""
    s = (s or "").strip().strip('"\''"”’")
    s = s.rstrip(string.punctuation + " ")
    return s


def _find_by_title_fuzzy(name: str, meta: List[Dict]) -> Optional[Dict]:
    """Find a method in meta by exact/contains/fuzzy match on title."""
    if not name: 
        return None
    name_lc = name.strip().lower()
    titles = [m.get("title","") for m in meta]

    # direct contains / exact
    for m in meta:
        t = (m.get("title","") or "").lower()
        if t == name_lc or t in name_lc or name_lc in t:
            return m

    # fuzzy best match
    match = difflib.get_close_matches(name, titles, n=1, cutoff=0.6)
    if match:
        for m in meta:
            if m.get("title","") == match[0]:
                return m
    return None

# =========================
# Phase & constraint heuristics
# =========================

phase_keywords: Dict[str, List[str]] ={
    "problem_framing": [
        "start","begin","early","initial","kickoff","kick-off","first step","how to start",
        "frame","define","explore","stakeholder","understand"
    ],
    "implementation": [
        "build","implement","prototype","develop","apply","execute","run","carry out","do it"
    ],
    "evaluation": [
        "evaluate","measure","feedback","assess","test","impact","results","review","monitor"
    ],
    "conclusion": [
        "end","ending","finish","finalize","wrap","wrap up","close","closure","conclude",
        "summarize","report","lessons","wind down"
    ],

}

def infer_phase_from_query(query: str) -> Optional[str]:
    q = (query or "").lower()
    for phase, kws in phase_keywords.items():
        if any(k in q for k in kws):
            return phase
    return None
def _phase_score(method: dict, phase: str) -> float:
    return float((method.get("phase_membership") or {}).get(phase, 0.0))

def _phase_mismatch_penalty(method: dict, desired_phase: str) -> float:
    pm = _phase_score(method, desired_phase)
    # penalize items that are really not in the asked phase
    if pm < 0.15: 
        return -0.35
    if pm < 0.30:
        return -0.18
    return 0.0

def _time_bucket_from_query(q: str) -> Optional[Tuple[int, int]]:
    q = (q or "").lower()
    if re.search(r"\b(quick|fast|short)\b", q) or any(w in q for w in ["10 min", "15 min", "under 30", "< 30"]):
        return (0, 30)
    if re.search(r"\bmedium\b", q):
        return (45, 180)
    if any(w in q for w in ["1 hour", "60 min", "~1h", "around an hour"]):
        return (45, 90)
    if any(w in q for w in ["half day", "half-day", "2-4 hours", "2 hours", "3 hours", "4 hours"]):
        return (120, 240)
    if any(w in q for w in ["full day", "1 day"]):
        return (360, 480)
    if "workshop" in q:
        return (120, 240)
    return None

def _resource_level_from_query(q: str) -> Optional[str]:
    q = (q or "").lower()
    if any(w in q for w in ["no budget", "low budget", "low resource", "lightweight", "paper", "pen"]):
        return "low"
    if any(w in q for w in ["moderate", "some tools", "medium"]):
        return "medium"
    if any(w in q for w in ["lab", "specialists", "expensive", "high budget"]):
        return "high"
    return None

def _group_size_from_query(q: str) -> Optional[Tuple[int, int]]:
    q = (q or "").lower()
    m = re.search(r"(\d+)\s*[-–—]\s*(\d+)\s*(people|participants|users)?\b", q)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.search(r"(\d+)\s*(people|participants|users)\b", q)
    if m:
        n = int(m.group(1))
        return (n, n)
    if re.search(r"\b(solo|individual|1 person)\b", q):
        return (1, 1)
    if "small group" in q:
        return (2, 6)
    if "large group" in q:
        return (7, 50)
    return None

def _facilitation_from_query(q: str) -> Optional[bool]:
    q = (q or "").lower()
    if any(w in q for w in ["no facilitator", "self guided", "self-guided", "async"]):
        return False
    if any(w in q for w in ["facilitator", "moderated", "guided"]):
        return True
    return None

def _must_include_terms(q: str) -> List[str]:
    quoted = [m.strip('"').lower() for m in re.findall(r'"([^"]+)"', q or "")]
    extras: List[str] = []
    if "stakeholder" in (q or "").lower():
        extras.append("stakeholder")
    if "power" in (q or "").lower():
        extras.append("power")
    return list(dict.fromkeys(quoted + extras))

def parse_constraints(query: str) -> Dict:
    return {
        "phase":        infer_phase_from_query(query),
        "time":         _time_bucket_from_query(query),
        "resources":    _resource_level_from_query(query),
        "group":        _group_size_from_query(query),
        "facilitation": _facilitation_from_query(query),
        "must_terms":   _must_include_terms(query),
    }

# =========================
# Card-field parsers & scoring
# =========================

def _minutes_from_card_time(card_time: str | None):
    if not card_time: 
        return None
    s = card_time.lower()

    m = re.search(rf'(\d+)\s*{_DASH}\s*(\d+)\s*(min|minute|h|hour)', s)
    if m:
        a, b, unit = int(m.group(1)), int(m.group(2)), m.group(3)
        mul = 1 if unit.startswith('m') else 60
        return (a*mul, b*mul)

    m = re.search(r'(\d+)\s*(min|minute|h|hour)', s)
    if m:
        a, unit = int(m.group(1)), m.group(2)
        mul = 1 if unit.startswith('m') else 60
        return (a*mul, a*mul)

    if 'workshop' in s:
        if 'half' in s:
            return (150, 240)
        if 'full' in s:
            return (360, 480)
        return (180, 300)

    m = re.search(rf'(\d+)\s*{_DASH}\s*(\d+)\s*day', s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (a*8*60, b*8*60)
    m = re.search(r'(\d+)\s*day', s)
    if m:
        a = int(m.group(1))
        return (a*8*60, a*8*60)

    m = re.search(rf'(\d+)\s*{_DASH}\s*(\d+)\s*week', s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (a*5*8*60, b*5*8*60)
    m = re.search(r'(\d+)\s*week', s)
    if m:
        a = int(m.group(1))
        return (a*5*8*60, a*5*8*60)

    if any(w in s for w in ['very short','quick','short']):   return (10, 45)
    if 'medium' in s:                                          return (45, 180)
    if any(w in s for w in ['long','full day']):               return (360, 600)
    return None

def _resource_level_from_card(card_res: str | None):
    if not card_res: 
        return None
    s = card_res.lower().replace('–','-').replace('—','-')
    if any(w in s for w in ['none','paper','post-it','pen','low']):
        return 'low'
    if any(w in s for w in ['low-medium','low - medium','medium','tool','software','moderate']):
        return 'medium'
    if any(w in s for w in ['high','lab','specialist','expensive']):
        return 'high'
    return None

def _group_from_card(card_group: str | None):
    if not card_group: 
        return None
    s = card_group.lower().replace('–','-').replace('—','-')
    m = re.search(rf'(\d+)\s*{_DASH}\s*(\d+)', s)
    if m: 
        return (int(m.group(1)), int(m.group(2)))
    m = re.search(r'(\d+)\s*(people|participants|users|experts?)', s)
    if m:
        n = int(m.group(1)); 
        return (n, n)
    if 'small' in s: return (2, 6)
    if 'large' in s: return (7, 50)
    return None

def _range_overlap(a: Optional[Tuple[int, int]], b: Optional[Tuple[int, int]]) -> float:
    if not a or not b:
        return 0.0
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    if hi <= lo:
        return 0.0
    width = max(a[1] - a[0], b[1] - b[0], 1)
    return (hi - lo) / width

def _contains_all(text: str, terms: List[str]) -> bool:
    t = (text or "").lower()
    return all(term in t for term in terms)

def score_constraints(method: Dict, c: Dict) -> float:
    """Soft score for how well a method fits parsed constraints."""
    score = 0.0

    if c.get("time"):
        overlap = _range_overlap(_minutes_from_card_time(method.get("time_cost")), c["time"])
        score += 0.8 * overlap

    if c.get("resources"):
        mres = _resource_level_from_card(method.get("resource_cost"))
        if mres == c["resources"]:
            score += 0.4

    if c.get("group"):
        overlap = _range_overlap(_group_from_card(method.get("group_size")), c["group"])
        score += 0.5 * overlap

    if c.get("facilitation") is not None and isinstance(method.get("facilitation_needed"), bool):
        score += 0.2 if (method["facilitation_needed"] == c["facilitation"]) else 0.0

    if c.get("must_terms"):
        bag = " ".join([
            method.get("title", ""),
            method.get("short_description", ""),
            method.get("long_description", ""),
            " ".join(method.get("keywords", [])),
        ])
        score += 0.6 if _contains_all(bag, c["must_terms"]) else 0.0

    return score

# =========================
# Data & FAISS index
# =========================

def _file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def load_data() -> List[Dict]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def build_or_load_index():
    """Return (faiss_index, meta_list). Rebuild if data.json changes."""
    data = load_data()
    current_hash = _file_hash(DATA_PATH)

    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(META_PATH) and os.path.exists(HASH_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        with open(HASH_PATH, "r", encoding="utf-8") as f:
            cached_hash = f.read().strip()

        if len(meta) == len(data) and cached_hash == current_hash:
            return index, meta
        # else fall through to rebuild

    texts, meta = [], []
    for m in data:
        texts.append(_doc_text(m))
        meta.append(m)

    embeddings = _model.encode(texts, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine via normalized dot product
    index.add(embeddings)

    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(HASH_PATH, "w", encoding="utf-8") as f:
        f.write(current_hash)

    return index, meta

# =========================
# Search
# =========================
def _phase_penalty(method: dict, inferred_phase: str | None) -> float:
    if not inferred_phase:
        return 0.0
    pm = method.get("phase_membership", {}) or {}
    here = float(pm.get(inferred_phase, 0.0))
    other_max = max([float(v) for k, v in pm.items() if k != inferred_phase] + [0.0])
    # penalize when asked phase is tiny and another phase dominates
    return 0.10 if (here < 0.25 and other_max >= 0.6) else 0.0
def keyword_boost(query: str, m: dict) -> float:
    q = (query or "").strip()
    text = " ".join([
        m.get("title",""), m.get("short_description",""), m.get("long_description",""),
        " ".join(m.get("keywords",[]) or []),
        " ".join(m.get("when_to_use",[]) or []),
    ]).lower()
    score = 0.0
    # quoted phrases
    for phrase in re.findall(r"\"([^\"]+)\"", q):
        if phrase.lower() in text:
            score += 0.18
    # light token nudge
    for tok in re.findall(r"[A-Za-z][A-Za-z\-]+", q):
        if tok.lower() in text:
            score += 0.02
    return min(score, 0.35)

def search_methods(
    query: str,
    index,
    meta: List[Dict],
    k: int = 8,
    fetch_k: Optional[int] = None,
    *,
    mode: str = "app",
    use_reranker: bool = False,
    alpha: float = 0.78,
    beta: float  = 0.14,
    gamma: float = 0.08,
    use_mmr: bool = False,
    mmr_lambda: float = 0.75,
    # NEW: small phase bias for “similar to …”
    similar_phase_bias: float = 0.0,   # recommend 0.10–0.20
) -> Tuple[List[Dict], Dict]:

    """
    Unified search with two behaviors:

    - mode="cosine": return pure FAISS cosine neighbors (no constraints/phase/penalties).
      This mirrors the 'Similar methods' UI panel. Optionally apply MMR.
    - mode="app":    return app-style ranking = FAISS similarity blended with
      soft constraint fit + phase prior (+ optional cross-encoder rerank).

    Returns: (results[:k], constraints_dict)
    """

    # ---------- Parse constraints & “similar to …” intent ----------
    constraints = parse_constraints(query)
    inferred_phase = constraints.get("phase")

    m = _SIMILAR_RE.search(query or "")
    similar_mode   = False
    target_title   = None
    seed_top_phase = None

    # default: use the raw user query text
    query_text = query

    if m:
        # “similar to X …”
        target_name = _clean_name(m.group(2))
        target = _find_by_title_fuzzy(target_name, meta)
        if target:
            similar_mode   = True
            target_title   = target.get("title", "")
            seed_top_phase = _dominant_phase(target.get("phase_membership", {}) or {})
            # use the seed’s document text as the query for cosine neighbors
            query_text = _doc_text(target)

        # if no target found, we just keep query_text = query (fallback)

    # Optional: let the user override the phase inside the text query
    if similar_mode:
        user_phase = constraints.get("phase")
        if user_phase:
            seed_top_phase = user_phase  # user instruction beats seed’s dominant phase

    # finally, make the vector ONCE
    query_vec = _model.encode([query_text], normalize_embeddings=True).astype("float32")

        
    # ---------- Retrieve a candidate pool ----------
    M = fetch_k or min(len(meta), max(k * 10, 100))
    scores, ids = index.search(query_vec, M)

    # ---------- Build candidate list with core signals ----------
    cands: List[Dict] = []
    for i, vec_score in zip(ids[0], scores[0]):
        mdoc = dict(meta[i])

        # In "similar to X", exclude the seed itself
        if similar_mode and target_title and mdoc.get("title") == target_title:
            continue

        faiss_score = float(vec_score)  # cosine sim because embeddings are normalized
    # --- Phase-aware boost for "similar to ..." (cosine mode) ---
        if mode == "cosine" and similar_mode and similar_phase_bias > 0.0 and seed_top_phase:
            pf = float((mdoc.get("phase_membership") or {}).get(seed_top_phase, 0.0))
            # Tiny penalty if it screams another phase (prevents off-phase dominators)
            other_max = max(
                [float(v) for k_, v in (mdoc.get("phase_membership") or {}).items()
                if k_ != seed_top_phase] + [0.0]
            )
            phase_pen = -0.05 if (pf < 0.20 and other_max >= 0.60) else 0.0
            mdoc["score"] = faiss_score + similar_phase_bias * pf + phase_pen
            mdoc["_faiss"] = faiss_score
            mdoc["_seed_top_phase"] = seed_top_phase
            mdoc["_phase_fit"] = pf
            mdoc["_phase_pen"] = phase_pen
            cands.append(mdoc)
            continue

        # ---- mode == "app": compute your existing priors/constraints ----
        # Constraints do not apply for explicit "similar to ..." queries
        cscore = 0.0 if similar_mode else score_constraints(mdoc, constraints)

        # Phase prior only if a phase is inferred and not in "similar" mode
        phase_fit = 0.0
        penalty = 0.0
        faiss_adj = faiss_score
        if (not similar_mode) and inferred_phase:
            phase_fit = _phase_score(mdoc, inferred_phase)
            penalty   = _phase_mismatch_penalty(mdoc, inferred_phase)
            # small direct lift for matching phase on the base similarity
            faiss_adj += 0.30 * phase_fit

        # Your previous mix preserved (as a "pre" score)
        pre = (0.75 * faiss_adj + 0.25 * cscore) + penalty

        mdoc["_faiss"] = faiss_score
        mdoc["_faiss_adj"] = faiss_adj
        mdoc["_constraints"] = cscore
        mdoc["_phase_fit"] = phase_fit
        mdoc["_phase_pen"] = penalty
        mdoc["_pre"] = pre
        cands.append(mdoc)

    # ---------- Fast path: cosine mode ----------
    if mode == "cosine":
        cands.sort(key=lambda r: r["score"], reverse=True)
        if use_mmr and cands:
            cands = mmr_diversify(cands, k=min(k, len(cands)), lambda_=mmr_lambda)
        return cands[:k], constraints

    # ---------- Optional cross-encoder rerank for app mode ----------
    if use_reranker and cands:
        ce_scores = _rerank_cross_encoder(query, cands, text_fn=_doc_text)
        # z-norm CE to stabilize blending
        import numpy as np
        ce = np.asarray(ce_scores, dtype="float32")
        if ce.std() > 1e-8:
            ce = (ce - ce.mean()) / ce.std()
        for mdoc, s_ce in zip(cands, ce.tolist()):
            mdoc["_ce"] = s_ce
            # Blend: CE (alpha) + pre (beta) + phase prior (gamma)
            mdoc["score"] = alpha * s_ce + beta * mdoc["_pre"] + gamma * mdoc["_phase_fit"]
    else:
        # No CE: just keep the pre-score
        for mdoc in cands:
            mdoc["score"] = mdoc["_pre"]

    # ---------- Final ordering (+ optional MMR) ----------
    cands.sort(key=lambda r: r["score"], reverse=True)
    if use_mmr and cands:
        cands = mmr_diversify(cands, k=min(k, len(cands)), lambda_=mmr_lambda)

    return cands[:k], constraints

# =========================
# Optional helpers
# =========================

def rank_phases(methods: List[Dict]):
    phases = ["problem_framing", "implementation", "evaluation", "conclusion"]
    phase_scores = {p: 0.0 for p in phases}
    for m in methods:
        for p in phases:
            phase_scores[p] += float(m.get("phase_membership", {}).get(p, 0.0)) * float(m.get("score", 1.0))
    return sorted(phase_scores.items(), key=lambda x: -x[1])

def generate_response_template(query: str, methods: List[Dict]) -> str:
    top_phases = rank_phases(methods)[:2]
    phase_names = " and ".join(p[0].replace("_", " ").title() for p in top_phases)
    titles = "\n".join([f"– **{m['title']}**" for m in methods])
    return f"Based on your question, these methods are commonly used in the *{phase_names}* phase:\n\n{titles}"
