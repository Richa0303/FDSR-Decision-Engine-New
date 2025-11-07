# app.py
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify, render_template, send_from_directory, session
import os, re, json, difflib, base64, html as _html
from datetime import datetime
from difflib import SequenceMatcher

# === Use your existing brain (FAISS + SBERT) ===
from search_engine import build_or_load_index, search_methods


# -----------------------------------------------------------------------------
# Smalltalk helpers
# -----------------------------------------------------------------------------
GREETING_RE = re.compile(
    r"^(hi|hello|hey|yo|hola|hallo|ciao|namaste|good\s*morn(ing)?|good\s*even(ing)?|good\s*after(noon)?)\b",
    re.I,
)
THANKS_RE = re.compile(r"\b(thanks|thank\s*you|tysm|thx)\b", re.I)
BYE_RE = re.compile(r"\b(bye|goodbye|see\s*ya|see\s*you|catch\s*you|ciao)\b", re.I)
# Follow-up questions about the last opened method
QA_RE = re.compile(
    r"(key\s*steps|steps|how\s*to|procedure|time|duration|resources|inputs|outputs|deliverables|"
    r"group\s*size|participants?|who\s*participates|roles?|facilitation|risks?|bias(?:es)?|"
    r"strengths|limitations?|prereq|pre-?requisites?|thought\s*styles?|"
    r"when\s*(to\s*use|should\s*it\s*be\s*appl(?:y|ied))|apply|appl(?:y|ied)|suitab(?:le|ility))",
    re.I,
)


EARLY_INTENT_RE = re.compile(r"\b(start|begin|early|kick[\s-]?off|first steps?)\b", re.I)
# add right after EARLY_INTENT_RE
STEPS_FOR_RE = re.compile(r"(steps?|how\s*to)\s+(for|in|of)\s+(.+)", re.I)
# Treat single bare tokens (likely names/junk) as off-topic
NAME_OR_NOISE_RE = re.compile(r"^[a-zA-Z][a-zA-Z\-']{2,24}$")

def is_probably_name_or_noise(q: str) -> bool:
    """
    Return True for inputs like 'richa', 'edy', 'john', etc.
    Single token, alphabetic/hyphen/apostrophe, not a known method title,
    and not obviously a phase tag.
    """
    if not q: 
        return True
    q = q.strip()
    # ignore your explicit phase commands
    if q.lower() in {"problem_framing","implementation","evaluation","conclusion","all"}:
        return False
    # more than one word -> probably meaningful, let it pass
    if len(q.split()) > 1:
        return False
    # single short token?
    if NAME_OR_NOISE_RE.match(q):
        # if it's actually a method title, let it pass
        titles_lc = { (m.get("title","") or "").lower() for m in methods_data }
        return q.lower() not in titles_lc
    return False

def mentions_method(query: str, title: str) -> bool:
    t = (title or "").lower()
    q = (query or "").lower()
    return t in q or difflib.SequenceMatcher(None, q, t).ratio() > 0.85

def field_text(m):
    parts = [
        m.get("title",""),
        m.get("short_description",""),
        m.get("long_description",""),
        " ".join(m.get("keywords",[]) or []),
        " ".join(m.get("when_to_use",[]) or []),
        " ".join(m.get("thought_styles",[]) or []),
        " ".join(m.get("participation",[]) or []),
    ]
    return " ".join([p for p in parts if p])

def keyword_boost(query, m):
    """Boost if quoted phrase(s) or bare tokens appear in text/keywords/title."""
    q = query.strip()
    text = field_text(m).lower()
    score = 0.0
    # quoted phrases
    for phrase in re.findall(r"\"([^\"]+)\"", q):
        if phrase.lower() in text:
            score += 0.18
    # individual tokens (light)
    for tok in re.findall(r"[A-Za-z][A-Za-z\-]+", q):
        if tok.lower() in text:
            score += 0.02
    return min(score, 0.35)

def phase_prior_boost(m, want_early=False):
    """Favor Problem Framing when user asks about starting."""
    if not want_early: return 0.0
    pm = m.get("phase_membership",{}) or {}
    # small bonus proportional to framing membership
    return 0.25 * float(pm.get("problem_framing", 0.0))

def cosine_like(a, b):
    """Very light textual similarity for MMR; good enough for short texts."""
    sa = set(re.findall(r"[A-Za-z]{3,}", field_text(a).lower()))
    sb = set(re.findall(r"[A-Za-z]{3,}", field_text(b).lower()))
    if not sa or not sb: return 0.0
    inter = len(sa & sb); uni = len(sa | sb)
    return inter / uni

def mmr_diversify(items, k=8, lambda_=0.75):
    """Maximal Marginal Relevance over current 'score' field."""
    if not items: return []
    selected = []
    cand = items[:]
    # seed with the best
    selected.append(cand.pop(0))
    while cand and len(selected) < k:
        best_i, best_val = 0, -1e9
        for i, it in enumerate(cand):
            rel = it["score"]
            div = max(cosine_like(it, s) for s in selected) if selected else 0.0
            val = lambda_ * rel - (1 - lambda_) * div
            if val > best_val:
                best_val, best_i = val, i
        selected.append(cand.pop(best_i))
    return selected


def detect_smalltalk(text: str):
    t = text.strip()
    if GREETING_RE.search(t):
        return "greeting"
    if THANKS_RE.search(t):
        return "thanks"
    if BYE_RE.search(t):
        return "bye"
    return None


def time_based_greeting():
    h = datetime.now().hour
    if 5 <= h < 12:
        return "Good morning"
    if 12 <= h < 17:
        return "Good afternoon"
    if 17 <= h < 23:
        return "Good evening"
    return "Hello"


def smalltalk_reply(kind: str) -> str:
    if kind == "greeting":
        return f"""
        <div class="suggested-methods-box">
          <div class="suggested-heading">👋 {time_based_greeting()}! Ask about any design method or phase, or try an example question below.</div>
        </div>
        """
    if kind == "thanks":
        return """
        <div class="suggested-methods-box">
          <div class="suggested-heading">🤗 You’re welcome! Let me know if you want to explore more methods or phases.</div>
        </div>
        """
    # bye
    return """
    <div class="suggested-methods-box">
      <div class="suggested-heading">👋 Bye! If you need a summary or export before leaving, just ask.</div>
    </div>
    """


# -----------------------------------------------------------------------------
# Flask app
# -----------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "richa12345")
# Build/load once for the whole app
INDEX, META = build_or_load_index()

# Load full methods for fuzzy & extra utilities (reuse search_engine metadata)
methods_data = META  # list[dict] from data.json
titles = [m.get("title", "") for m in methods_data]  # optional

# Serve your dataset to the frontend (for cards/graph)
@app.route("/data.json")
def serve_data_json():
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(backend_dir, "data.json")

titles = [m["title"] for m in methods_data]  # (unused, but handy for debugging)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def find_method_by_title_fuzzy(query: str):
    """Find a method when the user types the method name alone OR inside a sentence."""
    q = (query or "").strip().lower()
    if not q:
        return None

    # 1) Fast path: the method title appears inside the query
    for method in methods_data:
        title_l = method["title"].strip().lower()
        if title_l and (title_l == q or title_l in q):
            return method

    # 2) Heuristic: "... <name> method" pattern
    m = re.search(r"\b([a-z][a-z0-9 \-\/&]+)\s+method\b", q, re.I)
    if m:
        candidate = m.group(1).strip()
        all_titles = [m["title"] for m in methods_data]
        match = difflib.get_close_matches(candidate, all_titles, n=1, cutoff=0.5)
        if match:
            return next((m for m in methods_data if m["title"] == match[0]), None)

    # 3) Fallback: fuzzy against the whole query
    all_titles = [m["title"] for m in methods_data]
    match = difflib.get_close_matches(q, all_titles, n=1, cutoff=0.5)
    if match:
        return next((m for m in methods_data if m["title"] == match[0]), None)

    return None



def top_methods_by_phase(phase: str, threshold: float = 0.5):
    """Return methods strongly associated with a given phase."""
    result = []
    for m in methods_data:
        phase_membership = m.get("phase_membership", {})
        if phase == "conclusion":
            if phase_membership.get("conclusion", 0) >= threshold or m.get("phase") == "conclusion":
                result.append(m)
        else:
            if phase_membership.get(phase, 0) >= threshold:
                result.append(m)
    return sorted(result, key=lambda m: m.get("phase_membership", {}).get(phase, 0), reverse=True)


def render_method_links(titles_list, heading=None, heading_html=None):
    """Render a row of method buttons (used in list answers).
       If heading_html is provided, render it as a block above the buttons.
    """
    buttons = "".join(
        [f'<button class="method-button" onclick="sendMethod(\'{_html.escape(title)}\')">{_html.escape(title)}</button>'
         for title in titles_list]
    )

    if heading_html is not None:
        # Force a visible block above the buttons and add a hard break
        return (
            "<div class='bot-block'>"
            f"<div class='reply-heading' style='display:block;margin:0 0 8px 0;'>{heading_html}</div>"
            "<br/>"
            f"{buttons}"
            "</div>"
        )

    if heading:
        return f"<div class='bot-block'><strong>{heading}</strong><br>{buttons}</div>"

    return f"<div class='bot-block'>{buttons}</div>"


def _esc(s):
    return _html.escape(str(s if s is not None else ""))


def render_method_card(method: dict, similar_list=None):
    """
    Return a single, expandable method card faithful to TD-Net fields.
    Supports: short_description, long_description, steps, when_to_use, thought_styles,
    participation, outputs, prerequisites, deliverables, risks_biases, strengths, limitations,
     keywords, phase_membership, source, image, time/resource/group/facilitation (pills).
    """
    similar_list = similar_list or []
    PHASE_COLORS = {
        "problem_framing": "#3b82f6",
        "implementation": "#22c55e",
        "evaluation": "#f59e42",
        "conclusion": "#a855f7",
    }
    phase = method.get("phase") or ""
    phase_color = PHASE_COLORS.get(phase, "#6a4fa3")
    phase_label = phase.replace("_", " ").title() if phase else ""

    # Flexible steps (array or string)
    steps = method.get("steps") or method.get("key_steps", [])
    if isinstance(steps, str):
        steps_list_html = f"<p>{_esc(steps)}</p>"
    else:
        steps_list_html = "<ol>" + "".join(f"<li>{_esc(x)}</li>" for x in steps if x) + "</ol>"

    keywords = method.get("keywords", [])
    inputs = method.get("inputs", [])
    outputs = method.get("outputs", [])
    time_cost = method.get("time_cost")
    resource_cost = method.get("resource_cost")
    group_size = method.get("group_size")
    facil = method.get("facilitation_needed")

    risks = method.get("risks_biases", [])
    strengths = method.get("strengths", [])
    limitations = method.get("limitations", [])

    prereq = method.get("prerequisites", [])
    deliverables = method.get("deliverables", [])

    when_to_use = method.get("when_to_use", [])
    thought_styles = method.get("thought_styles", [])
    participation = method.get("participation", [])

    phase_membership = method.get("phase_membership", {})
    source = method.get("source")
    image = method.get("image")

    # --- small render helpers
    def pill(label, value):
        if value is None or value == "": return ""
        slug = str(label).strip().lower().replace(" ", "-")
        cls = f"pill pill-{slug}"
        return f'<span class="{cls}"><strong>{_esc(label)}:</strong> {_esc(value)}</span>'

    def list_block(label, arr):
        if not arr:
            return ""
        # chips row (short items)
        chips = "".join(f"<span class='chip'>{_esc(x)}</span>" for x in arr)
        return (
            f"<div class='method-section'>"
            f"<strong>{_esc(label)}:</strong>"
            f"<div class='chip-row'>{chips}</div>"
            f"</div>"
        )

    def list_block_bullets(label, arr, section_class: str = ""):
        if not arr:
            return ""
        items = "".join(f"<li>{_esc(x)}</li>" for x in arr)
        cls = f"method-section {section_class}".strip()
        return (
            f"<div class='{cls}'>"
            f"<strong>{_esc(label)}:</strong>"
            f"<ul class='nice-list'>{items}</ul>"
            f"</div>"
        )

    def chips_block(label, arr, chip_kind: str):
        if not arr:
            return ""
        chips = "".join(f"<span class='chip chip--{chip_kind}'>{_esc(x)}</span>" for x in arr)
        return (
            f"<div class='method-section'>"
            f"<strong>{_esc(label)}:</strong>"
            f"<div class='chip-row'>{chips}</div>"
            f"</div>"
        )

    # Phase membership display
    phase_html = ""
    if isinstance(phase_membership, dict) and phase_membership:
        phase_html = (
            '<div class="method-section"><strong>Phase membership:</strong> '
            + ", ".join(f"{_esc(k)}: {_esc(v)}" for k, v in phase_membership.items())
            + "</div>"
        )

    # Unknown fields -> extra block (keeps your “extra pills” spirit)
    known = {
        "title","short_description","long_description","steps","key_steps","score","keywords",
        "inputs","outputs","time_cost","resource_cost","group_size","facilitation_needed",
        "risks_biases","strengths","limitations","prerequisites","deliverables",
        "phase_membership","source","image","phase","when_to_use","thought_styles","participation"
    }
    extra_html = ""
    for k, v in method.items():
        if k in known:
            continue
        if v is None or (isinstance(v, (list, dict)) and not v) or (isinstance(v, str) and not v.strip()):
            continue
        label = k.replace("_", " ").title()
        if isinstance(v, list):
            if v and isinstance(v[0], (dict, list)):
                val = "".join(
                    f'<div style="margin-top:6px;padding-left:8px;">{_esc(json.dumps(item, ensure_ascii=False))}</div>'
                    for item in v
                )
            else:
                val = ", ".join(_esc(x) for x in v)
        elif isinstance(v, dict):
            val = f'<pre style="white-space:pre-wrap;margin:6px 0;padding:8px;background:#faf6ff;border-radius:8px;border:1px solid #efe7fb;">{_esc(json.dumps(v, ensure_ascii=False, indent=2))}</pre>'
        else:
            val = _esc(v)
        extra_html += f'<div class="method-section"><strong>{_esc(label)}:</strong> {val}</div>'

    # --- Similar methods (buttons only)
    similar_buttons = "".join(
        f"""
        <button class="method-button" onclick="sendMethod('{_esc(s.get("title",""))}')">
          {_esc(s.get("title",""))}
          <span class="sim-chip">sim {s.get("score",0):.2f}</span>
        </button>
        """
        for s in similar_list
    )

    similar_b64 = base64.b64encode(json.dumps(similar_list).encode("utf-8")).decode("ascii")
    visualize_btn = f"""
      <div style='text-align:center; margin-top: 18px;' class="visualize-row">
        <button class="graph-button"
                data-method="{_esc(method.get('title',''))}"
                data-similar-b64="{_esc(similar_b64)}">
          🔍 Visualize Similar Methods
        </button>
      </div>
    """
    feedback_html = f"""
      <div class='feedback-block' style="margin-top:12px;">
        Was this helpful?
        <button class='feedback-btn' data-feedback='helpful' data-method="{_esc(method.get('title',''))}">👍 Yes</button>
        <button class='feedback-btn' data-feedback='not_relevant' data-method="{_esc(method.get('title',''))}">👎 Not really</button>
      </div>
    """

    keywords_html = ""
    if keywords:
        keywords_html = (
            '<div class="method-section"><strong>Keywords:</strong>'
            '<div class="method-keywords">'
            + "".join(f'<span class="kw">{_esc(k)}</span>' for k in keywords)
            + "</div></div>"
        )

    meta_pills = "".join(
        filter(
            None,
            [
                pill("Time", time_cost),
                pill("Resources", resource_cost),
                pill("Group", group_size),
                pill("Facilitation", "Yes" if facil else "No") if facil is not None else "",
            ],
        )
    )
    meta_html = f'<div class="method-meta">{meta_pills}</div>' if meta_pills else ""

    image_html = (
        f'<img src="{_esc(image)}" alt="{_esc(method.get("title",""))}" class="node-card-img">' if image else ""
    )
    source_html = (
        f'<div class="method-section"><strong>Source:</strong> <a href="{_esc(source)}" target="_blank" style="color:#3b82f6;">{_esc(source)}</a></div>'
        if source
        else ""
    )
    phase_pill_html = f'<span class="phase-pill" style="background:{phase_color};">{_esc(phase_label)}</span>' if phase_label else ""

    # Compact meta + See-more button (keeps your UI behavior)
    compact_meta = meta_html or ""

    see_more_btn = """
<div class="see-more-row">
  <button class='see-more-btn'>See more</button>
</div>
"""

    full_details_html = f"""
<div class='method-card' style="--phase-color:{phase_color};">
  {image_html}
  <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
      <h3 class="node-card-title" style="margin:0;">{_esc(method.get('title',''))}</h3>
      {phase_pill_html}
  </div>

  <div class="node-card-desc" style="margin-top:8px;">
      <em>{_esc(method.get('short_description','')) or 'No description available.'}</em>
  </div>

  {compact_meta}
  {see_more_btn}

  <!-- Expanded content -->
  <div class='method-card-full' style="display:none;">

    {'<section class="long-desc"><strong>Description:</strong> ' + _esc(method.get('long_description','')) + '</section>' if method.get('long_description') else ''}

    {list_block_bullets('When should it be applied?', when_to_use, section_class='section--when')}
    {list_block_bullets('How does it work?', steps if isinstance(steps, list) else [steps] if steps else [], section_class='section--steps')}
    {list_block_bullets('How are thought styles bridged?', thought_styles, section_class='section--thought')}
    {list_block_bullets('Who participates in what role?', participation, section_class='section--participation')}

    {list_block_bullets('Inputs', inputs, section_class='section--inputs')}
    {list_block_bullets('Outputs', outputs, section_class='section--outputs')}
    {list_block_bullets('Risks / biases', risks, section_class='section--risks')}
    {list_block_bullets('Strengths', strengths, section_class='section--strengths')}
    {list_block_bullets('Limitations', limitations, section_class='section--limitations')}


    {list_block_bullets('Prerequisites', prereq, section_class='section--prereq')}

    {phase_html}
    {keywords_html}
    {source_html}

    {extra_html}

    <section class='method-section'>
      <strong>Similar Methods:</strong>
      <div class="similar-list" style="margin-top:6px;">{similar_buttons}</div>
      <div class="sim-explain overall" style="margin-top:8px; opacity:.8;">
        These methods share conceptual or practical similarities (keywords, phase profile, and description).
      </div>
    </section>

    {visualize_btn}
    {feedback_html}
  </div>
</div>
"""
    return full_details_html




def render_hint(text: str) -> str:
    return (
        "<div class='bot-block' "
        "style='background:#F6F2FB;border-left:4px solid #A78BFA;"
        "padding:10px 12px;border-radius:10px;margin:8px 0;'>"
        f"{_esc(text)}"
        "</div>"
    )

def maybe_prepend_hint(hint_text: str | None, dedupe_key: str | None = None) -> str:
    if not hint_text:
        return ""
    if dedupe_key:
        shown = set(session.get("shown_hints", []))
        if dedupe_key in shown:
            return ""
        shown.add(dedupe_key)
        session["shown_hints"] = list(shown)
    return render_hint(hint_text)

def render_all_phase_groups():
    html = ""
    phases = ["problem_framing", "implementation", "evaluation", "conclusion"]
    for phase in phases:
        methods = top_methods_by_phase(phase)
        if not methods:
            continue
        title_buttons = "".join(
            f'<button class="method-button" onclick="sendMethod(\'{_html.escape(m["title"])}\')">{_html.escape(m["title"])}</button>'
            for m in methods
        )
        html += (
            f'<div class="bot-block"><strong>Top methods in {phase.replace("_", " ").capitalize()} phase:</strong><br>'
            f'{title_buttons}</div>'
        )
    return html

def render_additional_details(details):
    if not details:
        return ""
    html_parts = []
    for key, value in details.items():
        title = key.replace("_", " ").capitalize()
        if isinstance(value, list):
            html_parts.append(
                f'<section><strong>{title}:</strong><ul>'
                + "".join(f"<li>{_esc(v)}</li>" for v in value)
                + "</ul></section>"
            )
        else:
            html_parts.append(f"<section><strong>{title}:</strong><p>{_esc(value)}</p></section>")
    return "".join(html_parts)


# -----------------------------------------------------------------------------
# Similarity helpers (TD-Net style)
# -----------------------------------------------------------------------------
def extract_method_name_from_query(query, phrases):
    query_lc = query.lower()
    for phrase in phrases:
        if phrase in query_lc:
            after_phrase = query_lc.split(phrase, 1)[-1].strip()
            after_phrase = after_phrase.strip("?.!")
            return after_phrase
    return None

def _keywords_set(m):
    return set(map(lambda s: s.lower().strip(), m.get("keywords", []) or []))

def _phase_vector(m):
    phs = ["problem_framing", "implementation", "evaluation", "conclusion"]
    pm = m.get("phase_membership", {}) or {}
    return [float(pm.get(p, 0.0)) for p in phs]

def fuzzy_similarity(method1, method2):
    """
    Similarity = 0.45 * keyword Jaccard
               + 0.35 * phase profile proximity
               + 0.20 * text similarity (short+long+steps)
    """
    # keywords overlap
    k1, k2 = _keywords_set(method1), _keywords_set(method2)
    if k1 or k2:
        jacc = len(k1 & k2) / max(1, len(k1 | k2))
    else:
        jacc = 0.0

    # phase profile (L1 distance → proximity)
    v1, v2 = _phase_vector(method1), _phase_vector(method2)
    if v1 or v2:
        l1 = sum(abs(a - b) for a, b in zip(v1, v2))
        phase_score = 1.0 - (l1 / 4.0)  # 4 dims normalized
    else:
        phase_score = 0.0

    # text similarity
    t1 = (method1.get("short_description","") + " " + method1.get("long_description","")).strip()
    t2 = (method2.get("short_description","") + " " + method2.get("long_description","")).strip()
    # include steps text if present
    s1 = method1.get("steps")
    s2 = method2.get("steps")
    if isinstance(s1, list): t1 += " " + " ".join(s1)
    elif isinstance(s1, str): t1 += " " + s1
    if isinstance(s2, list): t2 += " " + " ".join(s2)
    elif isinstance(s2, str): t2 += " " + s2
    text_score = SequenceMatcher(None, t1, t2).ratio() if (t1 and t2) else 0.0

    return 0.45 * jacc + 0.35 * phase_score + 0.20 * text_score

def get_fuzzy_similars(method, top_n=10):
    scored = []
    for m in methods_data:
        if m["title"] == method["title"]:
            continue
        score = fuzzy_similarity(method, m)
        scored.append({"title": m["title"], "score": score})
    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_n]

def generate_reason(method1, method2):
    # For now keep a generic reason (you can enrich later)
    k1, k2 = _keywords_set(method1), _keywords_set(method2)
    common = k1 & k2
    if common:
        return f"Both mention: {', '.join(sorted(common))}."
    return "These methods share conceptual or practical similarities."

def design_fallback_message(user_input):
    user_input_lc = user_input.lower()
    if any(
        word in user_input_lc
        for word in ["weather", "temperature", "sports", "news", "stock", "movie", "music", "joke", "recipe"]
    ):
        return """
        <div class="suggested-methods-box">
            <div class="suggested-heading">
                👋 Hi! I am a Design Insight Assistant.<br>
                I focus on design methods and phases.<br>
                <span style="display:block;margin-top:8px;">
                    Not sure what to ask? Select a question from the dropdown below or click a phase button to explore!
                </span>
            </div>
        </div>
        """
    return """
    <div class="suggested-methods-box">
        <div class="suggested-heading">
            👋 Hi! I am a Design Insight Assistant.<br>
            Ask me questions related to design methods and phases.<br>
            <span style="display:block;margin-top:8px;">
                Not sure what to ask? Select a question from the dropdown below!
            </span>
        </div>
    </div>
    """


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    method = data.get("method")
    feedback = data.get("feedback")
    timestamp = datetime.now().isoformat()
    log_entry = {"method": method, "feedback": feedback, "timestamp": timestamp}
    with open("feedback_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    return jsonify({"status": "ok"})


def get_methods_by_phases(phases, results=None, threshold=0.5):
    """Filter methods by multiple phases."""
    if results is None:
        results = methods_data
    filtered = []
    for m in results:
        for phase in phases:
            pm = m.get("phase_membership", {})
            if (pm.get(phase, 0) >= threshold) or (m.get("phase") == phase):
                filtered.append(m)
                break
    return filtered

def build_context_answer(method, user_input_lc):
    def _esc(s):
        return _html.escape(str(s) if s is not None else "")

    title = method.get("title", "this method")
    parts = []
    shown_sections = []  # track only sections that actually rendered anything

    def add_section(label, html_or_empty):
        """Append html if non-empty and record label for heading."""
        if html_or_empty:
            parts.append(html_or_empty)
            shown_sections.append(label)

    def fmt_list(label, arr):
        if not arr:
            return ""
        return (
            "<div style='margin:6px 0;'>"
            f"<strong>{_esc(label)}:</strong> "
            + ", ".join(_esc(x) for x in arr)
            + "</div>"
        )

    # --- Steps / procedure ---
    if re.search(r"(step|procedure|how\s*to)", user_input_lc):
        steps = method.get("steps") or method.get("key_steps")
        if isinstance(steps, list) and steps:
            html = (
                "<div style='margin:6px 0;'><strong>Steps:</strong>"
                "<ol style='margin:6px 0 0 18px;'>"
                + "".join(f"<li>{_esc(s)}</li>" for s in steps)
                + "</ol></div>"
            )
            add_section("Steps", html)
        elif isinstance(steps, str) and steps.strip():
            add_section("Steps", f"<p><strong>Steps:</strong> {_esc(steps)}</p>")
    # --- When to use / suitability ---
    if re.search(r"(when\s*(to\s*use|should)|apply|appl(?:y|ied)|suitab(?:le|ility)|use\s*cases?)", user_input_lc):
        wt = method.get("when_to_use", [])
        if wt:
            add_section("When should it be applied?",
                        "<div style='margin:6px 0;'><strong>When should it be applied?</strong>"
                        "<ul style='margin:6px 0 0 18px;'>"
                        + "".join(f"<li>{_esc(s)}</li>" for s in wt)
                        + "</ul></div>")
  

    if re.search(r"(thought\s*styles?|assumption|bridg)", user_input_lc):
            ts = method.get("thought_styles", [])
            if ts:
                html = "<div style='margin:6px 0;'><strong>How are thought styles bridged?</strong><ul style='margin:6px 0 0 18px;'>"
                html += "".join(f"<li>{_esc(x)}</li>" for x in ts)
                html += "</ul></div>"
                add_section("How are thought styles bridged?", html)

    # --- Participation / roles ---
    if re.search(r"(who\s*participates|participants?|roles?)", user_input_lc):
        part = method.get("participation", [])
        if part:
            add_section("Who participates in what role?",
                        "<div style='margin:6px 0;'><strong>Who participates in what role?</strong>"
                        "<ul style='margin:6px 0 0 18px;'>"
                        + "".join(f"<li>{_esc(s)}</li>" for s in part)
                        + "</ul></div>")

    # --- Time ---
    if re.search(r"(time|duration|how\s*long)", user_input_lc):
        tc = method.get("time_cost")
        if tc:
            add_section("Time", f"<p><strong>Time:</strong> {_esc(tc)}</p>")

    # --- Resources / Inputs ---
    if re.search(r"(resource|materials|tools|inputs?)", user_input_lc):
        rc_html = ""
        rc = method.get("resource_cost")
        if rc:
            rc_html += f"<p><strong>Resources:</strong> {_esc(rc)}</p>"
        inputs_html = fmt_list("Inputs", method.get("inputs", []))
        add_section("Resources" if rc else "", rc_html)             # may be ""
        add_section("Inputs" if inputs_html else "", inputs_html)    # may be ""

    # --- Outputs / Deliverables ---
    if re.search(r"(output|deliverable|result)", user_input_lc):
        out_html = fmt_list("Outputs", method.get("outputs", []))
        deliv_html = fmt_list("Deliverables", method.get("deliverables", []))
        add_section("Outputs" if out_html else "", out_html)
        add_section("Deliverables" if deliv_html else "", deliv_html)

    # --- Group size ---
    if re.search(r"(group|participants|people)", user_input_lc):
        gs = method.get("group_size")
        if gs:
            add_section("Group size", f"<p><strong>Group size:</strong> {_esc(gs)}</p>")

    # --- Facilitation ---
    if re.search(r"(facilitation|facilitator|moderation)", user_input_lc):
        fac = method.get("facilitation_needed")
        if fac is not None:
            add_section("Facilitation", f"<p><strong>Facilitation needed:</strong> {'Yes' if fac else 'No'}</p>")

    # --- Risks / strengths / limitations / suitability / prereqs ---
    if re.search(r"(risk|bias)", user_input_lc):
        rb = fmt_list("Risks / biases", method.get("risks_biases", []))
        add_section("Risks / biases" if rb else "", rb)

    if re.search(r"(strength|benefit|pros?)", user_input_lc):
        st = fmt_list("Strengths", method.get("strengths", []))
        add_section("Strengths" if st else "", st)

    if re.search(r"(limitation|weakness|cons?)", user_input_lc):
        lm = fmt_list("Limitations", method.get("limitations", []))
        add_section("Limitations" if lm else "", lm)

    if re.search(r"(prereq|pre-?requisite|preparation)", user_input_lc):
        pr = fmt_list("Prerequisites", method.get("prerequisites", []))
        add_section("Prerequisites" if pr else "", pr)

    # Nothing matched/rendered
    if not parts:
        return ("I didn’t find details for that question on this method—try asking about steps, time, "
                "resources, group size, outputs, risks, strengths, or limitations.")

    # Build heading text from sections actually shown (dedup while preserving order)
    seen = set()
    shown_sections = [s for s in shown_sections if s and not (s in seen or seen.add(s))]

    if len(shown_sections) == 0:
        section = "Details"
    elif len(shown_sections) == 1:
        section = shown_sections[0]
    elif len(shown_sections) == 2:
        section = f"{shown_sections[0]} & {shown_sections[1]}"
    else:
        section = f"{', '.join(shown_sections[:-1])} & {shown_sections[-1]}"

    # After computing `section` and before building the final HTML:
    short_desc = method.get('short_description', '') or ''
    short_desc_html = f"""
    <div class="node-card-subwrap">
      <span class="node-card-sub-bullet" aria-hidden="true"></span>
      <p class="node-card-sub">{_esc(short_desc)}</p>
    </div>
    """

    heading_html = (
        "<div style='font-weight:700;margin:2px 0 10px 0;'>"
        f"📌 {section} for <b>{_esc(title)}</b>"
        "</div>"
    )
        # Safe JS string for the title
    title_js = json.dumps(method.get("title", ""))

       # Open-full-method button (robust to quotes via data-title)
    open_btn = f"""
    <div class="see-more-row" style="text-align:right;margin-top:12px;">
      <button class="see-more-btn"
              data-title="{_esc(title)}"
              onclick="sendMethod(this.dataset.title)">
        🔎 Open full method
      </button>
    </div>
    """

    # Final block (note: we append the button at the end)
    return "<div class='bot-block'>" + heading_html + short_desc_html + "".join(parts) + open_btn + "</div>"




@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()

    # --- Smalltalk first
    kind = detect_smalltalk(user_input)
    if kind:
        return jsonify({"reply": smalltalk_reply(kind)})

    user_input_lc = user_input.lower()
    # Off-topic / single-token guard (e.g., "richa", "edy")
    if is_probably_name_or_noise(user_input):
        return jsonify({"reply": render_phase_picker("Click a phase to explore relevant methods:")})


    # --- Context Q&A about the *last opened* method (steps/time/etc.) ---
    if QA_RE.search(user_input_lc):
        # Prefer an explicitly named method (including "steps for X")
        explicit_method = find_method_by_title_fuzzy(user_input)
        if not explicit_method:
            m = STEPS_FOR_RE.search(user_input)
            if m:
                explicit_method = find_method_by_title_fuzzy(m.group(3))

        if explicit_method:
            session["last_method_title"] = explicit_method.get("title", "")
            html = build_context_answer(explicit_method, user_input_lc)
            return jsonify({"reply": html})

        # Only use last opened method if the query actually mentions it
        last_title = session.get("last_method_title")
        if last_title and mentions_method(user_input, last_title):
            method_ctx = next((m for m in methods_data if m["title"].lower() == last_title.lower()), None)
            if method_ctx:
                html = build_context_answer(method_ctx, user_input_lc)
                return jsonify({"reply": html})
        # otherwise fall through to normal search

    # Direct phase buttons (problem_framing / implementation / evaluation / conclusion / all)
    valid_phases = ["problem_framing", "implementation", "evaluation", "conclusion", "all"]
    if user_input_lc in valid_phases:
        if user_input_lc == "all":
            return jsonify({"reply": render_all_phase_groups()})

        # fetch methods for this single phase
        top_methods = top_methods_by_phase(user_input_lc)
        titles_list = [m["title"] for m in top_methods]

        # build a proper heading for this exact phase
        phase_key = user_input_lc
        found_list = [phase_key] if phase_key in PHASE_LABELS else []
        heading_html = build_heading(
            found_list,
            constraints=None,
            query=f"{PHASE_LABELS.get(phase_key, phase_key.replace('_',' ').title())} phase",
            total=len(top_methods),
        )

        return jsonify({"reply": render_method_links(titles_list, heading_html=heading_html)})

    # “methods like … / similar to … / …” intent
    similar_phrases = ["methods like", "similar to", "related to", "alternatives to"]
    if any(phrase in user_input_lc for phrase in similar_phrases):
        method_name = extract_method_name_from_query(user_input, similar_phrases)
        method = find_method_by_title_fuzzy(method_name) if method_name else None
        if method:
            similar = get_fuzzy_similars(method, top_n=6)
            similar_titles = [s["title"] for s in similar]
            heading = f'Similar methods to <b>{_esc(method["title"])}</b> are:'
            return jsonify({"reply": render_method_links(similar_titles, heading=heading)})
        else:
            return jsonify(
                {"reply": "🤔 Sorry, I couldn't find a method matching your query. Please check the method name or try another."}
            )

    # “start of process” intent (Problem Framing + Implementation) — green heading
    start_keywords = [
        "start", "begin", "kickoff", "kick-off", "kick off", "first step", "first steps",
        "early stage", "where to begin", "how do i start", "how to start"
    ]
    if any(kw in user_input_lc for kw in start_keywords):
        # 1) Pull early-phase methods
        start_phases = ["problem_framing", "implementation"]
        start_methods = get_methods_by_phases(start_phases, threshold=0.6)

        # 2) Apply your existing post-search boosts
        want_early = True  # enables phase_prior_boost toward Problem Framing
        for r in start_methods:
            base = float(r.get("score", 0.0))
            kb = float(keyword_boost(user_input, r))
            pb = float(phase_prior_boost(r, want_early))
            r["score"] = min(1.0, base + kb + pb)

        # 3) Sort + diversify to avoid near-duplicates
        start_methods.sort(key=lambda m: m.get("score", 0.0), reverse=True)
        start_methods = mmr_diversify(start_methods, k=min(12, len(start_methods)), lambda_=0.75)

        # 4) Heading + one-time hint + list
        green = "#22c55e"
        blue = "#1976D2"
        heading = (
            "🔎 For the start of the design process, these methods from the "
            f"<span style='color:{blue};font-weight:bold;'>Problem Framing</span> and "
            f"<span style='color:{green};font-weight:bold;'>Implementation</span> phases are commonly used:"
        )

        titles_list = [m["title"] for m in start_methods]
        start_hint = "At the start: clarify actors, goals, constraints; co-define scope and success."
        hint_html  = maybe_prepend_hint(start_hint, dedupe_key="hint:start_block")

        reply_html = hint_html + render_method_links(titles_list, heading=heading)
        return jsonify({"reply": reply_html})

    # “end of process” intent (evaluation + conclusion)
    end_keywords = [
        "end",
        "final",
        "finish",
        "last part",
        "complete",
        "closing",
        "at the end",
        "end of",
        "finish up",
        "finalize",
    ]

    if any(kw in user_input_lc for kw in end_keywords):
        # 1) Pull late-phase methods
        end_phases = ["evaluation", "conclusion"]
        end_methods = get_methods_by_phases(end_phases, threshold=0.5)

        # 2) Apply post-search boosts (no early prior here)
        want_early = False
        for r in end_methods:
            base = float(r.get("score", 0.0))
            kb   = float(keyword_boost(user_input, r))
            pb   = float(phase_prior_boost(r, want_early))  # will be 0.0
            r["score"] = min(1.0, base + kb + pb)

        # 3) Sort + diversify
        end_methods.sort(key=lambda m: m.get("score", 0.0), reverse=True)
        end_methods = mmr_diversify(end_methods, k=min(12, len(end_methods)), lambda_=0.75)

        # 4) Heading + list
        yellow = "#fbc02d"   # Evaluation
        red    = "#d32f2f"   # Conclusion
        heading = (
            "🔎 For the end of the design process, these methods from the "
            f"<span style='color:{yellow};font-weight:bold;'>Evaluation</span> and "
            f"<span style='color:{red};font-weight:bold;'>Conclusion</span> phases are commonly used:"
        )

        titles_list = [m["title"] for m in end_methods]
        return jsonify({"reply": render_method_links(titles_list, heading=heading)})

    # Natural-language hints to phases
    natural_questions = {
        "problem_framing": [
            "frame a design problem",
            "understand stakeholders",
            "problem framing",
            "what methods help frame",
            "identify needs",
            "early stage",
        ],
        "implementation": [
            "implement an idea",
            "make a prototype",
            "execute",
            "carry out the plan",
            "run the solution",
            "implementation techniques",
        ],
        "evaluation": [
            "evaluate a prototype",
            "feedback",
            "measure success",
            "assessment",
            "testing",
            "evaluate",
            "review",
        ],
        "conclusion": [
            "conclude",
            "wrap up",
            "document learnings",
            "what was learned",
            "summarize",
            "report",
            "close the project",
        ],
    }
    for phase, keywords in natural_questions.items():
        if any(kw in user_input_lc for kw in keywords):
            top_methods = top_methods_by_phase(phase)
            titles_list = [m["title"] for m in top_methods]
            return jsonify({
                "reply": render_method_links(
                    titles_list,
                    heading=f"🧩 Suggested methods for <b>{phase.replace('_', ' ').capitalize()}</b> phase:"
                )
            })

    # --- Direct method card if the input looks like a title (NOW before semantic search) ---
    method = find_method_by_title_fuzzy(user_input)
    if method:
        session["last_method_title"] = method.get("title", "")
        similar = []
        for s in get_fuzzy_similars(method, top_n=10):
            sm = find_method_by_title_fuzzy(s["title"])
            similar.append({
                "title": s["title"],
                "score": s["score"],
                "reason": generate_reason(method, sm) if sm else "",
                "phase": (sm or {}).get("phase"),
                "image": (sm or {}).get("image"),
            })
        return jsonify({"reply": render_method_card(method, similar)})

    # === Unified brain: semantic search (+ optional constraints)
    constraints = None
    try:
        out = search_methods(user_input, INDEX, META)
        if isinstance(out, tuple) and len(out) == 2:
            results, constraints = out
        else:
            results = out
    except Exception:
        results, constraints = [], None

    if results:
        # 1) Apply boosts BEFORE any low-confidence fallback
        want_early = bool(EARLY_INTENT_RE.search(user_input_lc))
        for r in results:
            base = float(r.get("score", 0.0))
            kb   = float(keyword_boost(user_input, r))
            pb   = float(phase_prior_boost(r, want_early))
            r["score"] = min(1.0, base + kb + pb)

        # 2) Re-rank & diversify
        results.sort(key=lambda m: m.get("score", 0.0), reverse=True)
        results = mmr_diversify(results, k=min(8, len(results)), lambda_=0.75)

        # 3) NOW do the low-confidence check on boosted score
        top = results[0].get("score", 0.0)
        second = results[1].get("score", 0.0) if len(results) > 1 else 0.0
        if (top < 0.50) or (top - second < 0.06):
    
            phase_pills = """
            <div style="font-size:17px;color:#6a4fa3;font-weight:600;margin-bottom:10px;">
              Welcome to Design Science Research! Which phase are you in currently?
            </div>
            <div style="display:flex;gap:12px;flex-wrap:wrap;margin:10px 0 2px 0;">
              <button class="phase-btn" onclick="sendTag('problem_framing')">Problem Framing</button>
              <button class="phase-btn" onclick="sendTag('implementation')">Implementation</button>
              <button class="phase-btn" onclick="sendTag('evaluation')">Evaluation</button>
              <button class="phase-btn" onclick="sendTag('conclusion')">Conclusion</button>
            </div>
            <div style="margin-top:8px;font-size:15px;color:#6a4fa3;font-weight:500;">
              (Click a phase above to explore relevant methods.)
            </div>
            """
            return jsonify({"reply": phase_pills})

        # 4) Build the compact, styled heading line using your helper
        found_phases = set()
        for m in results:
            p = m.get("phase")
            if p in PHASE_LABELS:
                found_phases.add(p)
            pm = m.get("phase_membership", {})
            for ph in PHASE_LABELS:
                if pm.get(ph, 0) >= 0.5:
                    found_phases.add(ph)

        found_phases = list(found_phases)
        if EARLY_INTENT_RE.search(user_input_lc):
            found_phases = [p for p in found_phases if p in ("problem_framing", "implementation")]

        heading_html = build_heading(
            found_phases,
            constraints,
            query=user_input,
            total=len(results)
        )

        titles_list = [m["title"] for m in results]
        return jsonify({"reply": render_method_links(titles_list, heading_html=heading_html)})

    # Final fallback
    return jsonify({"reply": design_fallback_message(user_input)})


# put this helper near your other small render helpers
def _pill(label, value):
    if value is None or value == "": return ""
    return f'<span class="pill pill-dark" style="margin-right:6px;">{label}: {value}</span>'
def render_phase_picker(message: str | None = None) -> str:
    """
    Purple hint + 4 phase buttons that call sendTag('<phase>').
    """
    msg = message or (
        "Welcome to Design Science Research! Click a phase below to explore relevant methods."
    )
    return f"""
    <div class="suggested-methods-box">
      <div class="suggested-heading" style="font-size:17px;color:#6a4fa3;font-weight:600;margin-bottom:10px;">
        {_esc(msg)}
      </div>
      <div style="display:flex;gap:12px;flex-wrap:wrap;margin:10px 0 2px 0;">
        <button class="phase-btn" onclick="sendTag('problem_framing')">Problem Framing</button>
        <button class="phase-btn" onclick="sendTag('implementation')">Implementation</button>
        <button class="phase-btn" onclick="sendTag('evaluation')">Evaluation</button>
        <button class="phase-btn" onclick="sendTag('conclusion')">Conclusion</button>
      </div>
      <div style="margin-top:8px;font-size:15px;color:#6a4fa3;font-weight:500;">
        (Click a phase above to explore relevant methods.)
      </div>
    </div>
    """

def _format_minutes_range(rng):
    if not rng: return None
    a, b = rng
    # show as 45–180m or 1–3h if large
    if b >= 120 and a % 60 == 0 and b % 60 == 0:
        return f"{a//60}–{b//60}h"
    return f"{a}–{b}m"

PHASE_LABELS = {
    "problem_framing": "Problem Framing",
    "implementation": "Implementation",
    "evaluation": "Evaluation",
    "conclusion": "Conclusion",
}

def build_heading(found_phases, constraints, *, query=None, total=None):
    # phase_text is currently unused; you can keep or remove it
    phase_text = ", ".join(PHASE_LABELS[p] for p in found_phases) if found_phases else "your query"

    pills = []
    if constraints:
        t = _format_minutes_range(constraints.get("time"))
        if t: pills.append(_pill("Time", t))
        res = constraints.get("resources")
        if res: pills.append(_pill("Resources", res))
        grp = constraints.get("group")
        if grp: pills.append(_pill("Group", f"{grp[0]}–{grp[1]}"))
        fac = constraints.get("facilitation")
        if fac is not None: pills.append(_pill("Facilitation", "Yes" if fac else "No"))

    count_html = f"<span style='opacity:.8;font-weight:600;'>({total} found)</span>" if total is not None else ""
    q_html = f" for <i>“{_esc(query)}”</i>" if query else ""

    return (
        "<div class='reply-heading-inner' style='display:block;font-weight:700;margin-bottom:6px;'>"
        f"🔎 Suggested methods{q_html} {count_html}"
        "</div>"
        f"<div class='reply-pills' style='margin:6px 0 10px 0;'>{''.join(pills)}</div>"
    ).strip()


# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
