# similar_engine.py
# -*- coding: utf-8 -*-
import json, os, re, difflib
from difflib import SequenceMatcher

DATA_PATH = os.path.join(os.path.dirname(__file__), "data.json")

def _load_methods():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

_METHODS = _load_methods()
_TITLE2METHOD = {m.get("title",""): m for m in _METHODS}
_TITLES = list(_TITLE2METHOD.keys())

def _keywords_set(m): 
    return set(s.lower().strip() for s in (m.get("keywords") or []))

def _phase_vector(m):
    phs = ["problem_framing", "implementation", "evaluation", "conclusion"]
    pm = m.get("phase_membership", {}) or {}
    return [float(pm.get(p, 0.0)) for p in phs]

def _fuzzy_similarity(m1, m2):
    # --- keyword Jaccard
    k1, k2 = _keywords_set(m1), _keywords_set(m2)
    jacc = (len(k1 & k2) / max(1, len(k1 | k2))) if (k1 or k2) else 0.0
    # --- phase proximity (1 - L1/4)
    v1, v2 = _phase_vector(m1), _phase_vector(m2)
    if v1 or v2:
        l1 = sum(abs(a-b) for a,b in zip(v1, v2))
        phase_score = 1.0 - (l1/4.0)
    else:
        phase_score = 0.0
    # --- text similarity (short+long+steps)
    t1 = (m1.get("short_description","") + " " + m1.get("long_description","")).strip()
    t2 = (m2.get("short_description","") + " " + m2.get("long_description","")).strip()
    s1, s2 = m1.get("steps"), m2.get("steps")
    if isinstance(s1, list): t1 += " " + " ".join(s1)
    elif isinstance(s1, str): t1 += " " + s1
    if isinstance(s2, list): t2 += " " + " ".join(s2)
    elif isinstance(s2, str): t2 += " " + s2
    text_score = SequenceMatcher(None, t1, t2).ratio() if (t1 and t2) else 0.0

    return 0.45*jacc + 0.35*phase_score + 0.20*text_score

def similar_titles(seed_title: str, top_n: int = 10):
    seed = _TITLE2METHOD.get(seed_title)
    if not seed:
        # mild fuzzy find if title not exact
        match = difflib.get_close_matches(seed_title, _TITLES, n=1, cutoff=0.5)
        if not match:
            return []
        seed = _TITLE2METHOD[match[0]]

    scored = []
    for m in _METHODS:
        if m.get("title") == seed.get("title"):
            continue
        s = _fuzzy_similarity(seed, m)
        scored.append((m.get("title",""), s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [t for (t, s) in scored[:top_n]]
