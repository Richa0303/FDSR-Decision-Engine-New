import json, re, copy
from urllib.parse import urlparse

CANON_LIST_FIELDS = [
    "keywords","when_to_use","thought_styles","participation","inputs","outputs",
    "prerequisites","deliverables","risks_biases","strengths","limitations","steps"
]
PHASE_KEYS = ["problem_framing","implementation","evaluation","conclusion"]

def _as_list(x):
    if x is None: return []
    if isinstance(x, list): return [str(i).strip() for i in x if str(i).strip()]
    if isinstance(x, str) and x.strip(): return [x.strip()]
    return []

def _https(url):
    if not url: return url
    u = url.strip()
    if not u: return None
    if not re.match(r"^https?://", u, re.I):
        u = "https://" + u
    return u

def normalize_method(m):
    m = copy.deepcopy(m)

    # Merge synonyms into canonical
    # why_apply/why_use/suitable_for -> when_to_use
    merged_when = []
    for k in ("when_to_use","why_apply","why_use","suitable_for"):
        merged_when += _as_list(m.get(k))
    m["when_to_use"] = merged_when

    # Steps -> list
    m["steps"] = _as_list(m.get("steps") or m.get("key_steps"))

    # Ensure list fields exist
    for fld in CANON_LIST_FIELDS:
        m[fld] = _as_list(m.get(fld))

    # Phase membership
    pm = m.get("phase_membership") or {}
    pm2 = {}
    for k in PHASE_KEYS:
        v = float(pm.get(k, 0.0) or 0.0)
        pm2[k] = max(0.0, min(1.0, v))
    m["phase_membership"] = pm2

    # URLs
    if m.get("source"):
        m["source"] = _https(m["source"])
    if m.get("image"):
        m["image"] = _https(m["image"])

    # Keywords default from title tokens if empty (light fallback)
    if not m["keywords"]:
        title = (m.get("title") or "").lower()
        toks = [t for t in re.findall(r"[a-z]{3,}", title) if t not in {"method","approach","tool"}]
        m["keywords"] = toks[:5]

    return m

def dedupe_by_title(items):
    seen = {}
    for m in items:
        t = (m.get("title") or "").strip()
        if not t: continue
        if t not in seen:
            seen[t] = m
        else:
            # merge arrays conservatively
            base = seen[t]
            for fld in CANON_LIST_FIELDS:
                base[fld] = list(dict.fromkeys(base.get(fld, []) + m.get(fld, [])))
            # prefer non-empty source/time/group/etc.
            for fld in ["source","time_cost","resource_cost","group_size"]:
                if not base.get(fld) and m.get(fld):
                    base[fld] = m[fld]
            # keep higher phase memberships
            for k in PHASE_KEYS:
                base["phase_membership"][k] = max(
                    base["phase_membership"].get(k,0.0),
                    (m.get("phase_membership") or {}).get(k,0.0)
                )
    return list(seen.values())

if __name__ == "__main__":
    with open("data.json","r",encoding="utf-8") as f:
        raw = json.load(f)
    norm = [normalize_method(m) for m in raw]
    norm = dedupe_by_title(norm)
    with open("data.normalized.json","w",encoding="utf-8") as f:
        json.dump(norm, f, ensure_ascii=False, indent=2)
    print(f"Normalized {len(raw)} → {len(norm)} methods into data.normalized.json")
