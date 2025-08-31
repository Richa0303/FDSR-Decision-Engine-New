from flask import Flask, request, jsonify, render_template
import json
import difflib
from datetime import datetime
from semantic_sim import load_methods, build_embeddings, get_similar_methods

app = Flask(__name__)

# Load method data + embeddings
methods_data = load_methods()
titles, embeddings = build_embeddings(methods_data)

def find_method_by_title_fuzzy(query):
    query_lower = query.strip().lower()
    for method in methods_data:
        if query_lower in method["title"].lower():
            return method
    all_titles = [m["title"] for m in methods_data]
    match = difflib.get_close_matches(query, all_titles, n=1, cutoff=0.5)
    if match:
        return next((m for m in methods_data if m["title"] == match[0]), None)
    return None

def top_methods_by_phase(phase, threshold=0.5):
    return sorted(
        [m for m in methods_data if m.get("phase_membership", {}).get(phase, 0) >= threshold],
        key=lambda m: m["phase_membership"][phase],
        reverse=True
    )

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    method = data.get("method")
    feedback = data.get("feedback")
    timestamp = datetime.now().isoformat()
    log_entry = {
        "method": method,
        "feedback": feedback,
        "timestamp": timestamp
    }
    with open("feedback_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    return jsonify({"status": "ok"})

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip().lower()

    valid_phases = ["problem_framing", "implementation", "evaluation", "conclusion", "all"]
    if user_input in valid_phases:
        if user_input == "all":
            return jsonify({"reply": render_all_phase_groups()})
        top_methods = top_methods_by_phase(user_input)
        titles_list = [m["title"] for m in top_methods]
        return jsonify({"reply": render_method_links(titles_list, heading=f"Top methods in {user_input.replace('_', ' ').capitalize()} phase:")})

    # Handle similarity search
    for phrase in ["methods like", "similar to", "related to", "alternatives to"]:
        if phrase in user_input:
            for method in methods_data:
                if method["title"].lower() in user_input:
                    index = titles.index(method["title"])
                    similar = get_similar_methods(index, titles, embeddings, top_n=6, return_scores=True)
                    return jsonify({
                        "reply": render_method_card(method, similar)
                    })

    # Handle method lookup
    method = find_method_by_title_fuzzy(user_input)
    if method:
        index = titles.index(method['title'])
        similar = get_similar_methods(index, titles, embeddings, top_n=6, return_scores=True)
        return jsonify({
            "reply": render_method_card(method, similar)
        })

    return jsonify({"reply": "Sorry, I couldn’t find that method. Try again or click a phase below."})

# -- HTML generators --

def render_method_card(method, similar_list=None):
    similar_buttons = ''.join([
    f'<button class="method-button" onclick="sendMethod(\'{s["title"]}\')">{s["title"]} (sim {s["score"]:.2f})</button>'
    for s in similar_list
])
    similar_html = f"""
    <section><strong>Similar Methods:</strong>
    <div class="similar-buttons-group">
        {similar_buttons}
    </div>
    </section>
    """


    visualize_btn = f"""
    <div style='text-align: center; margin-top: 20px;'>
      <button class="graph-button"
              data-method="{method['title']}"
              data-similar='{json.dumps(similar_list).replace("'", "&quot;")}'>
        🔍 Visualize Similar Methods
      </button>
    </div>
    """

    feedback_html = f"""
    <div class='feedback-block'>
      Was this helpful?
      <button class='feedback-btn' onclick="sendFeedback('helpful', '{method['title']}')">👍 Yes</button>
      <button class='feedback-btn' onclick="sendFeedback('not_relevant', '{method['title']}')">👎 Not really</button>
    </div>
    """

    description_html = f"""
    <div class='method-card'>
      <h3>{method['title']}</h3>
      <em>{method.get('short_description', '')}</em>
      <section><strong>Description:</strong><p>{method.get('long_description', '')}</p></section>
      <span class='expand-toggle' onclick='toggleExpand(this)'>Learn more...</span>
      <div class='expandable' style='display:none;'>
        <section><strong>Steps:</strong><ul>{''.join(f'<li>{step}</li>' for step in method.get('steps', []))}</ul></section>
        <span class='second-expand-toggle' onclick='toggleInnerExpand(this)'>More info...</span>
        <div class='second-expandable' style='display:none;'>
          {render_additional_details(method.get("details", {}))}
        </div>
        <section><strong>Source:</strong> <a href="{method.get("source", "#")}" target="_blank">{method.get("source", "")}</a></section>
        <section><strong>Similar Methods:</strong><br>{similar_buttons}</section>
        {visualize_btn}
        {feedback_html}
      </div>
    </div>
    """
    return description_html

def render_additional_details(details):
    if not details:
        return "<p>No additional details provided.</p>"
    html_parts = []
    for key, value in details.items():
        title = key.replace("_", " ").capitalize()
        if isinstance(value, list):
            html_parts.append(f'<section><strong>{title}:</strong><ul>' + ''.join(f'<li>{v}</li>' for v in value) + '</ul></section>')
        else:
            html_parts.append(f'<section><strong>{title}:</strong><p>{value}</p></section>')
    return ''.join(html_parts)

def render_method_links(titles, heading="All available methods:"):
    buttons = "".join(
        [f'<button class="method-button" onclick="sendMethod(\'{title}\')">{title}</button>' for title in titles]
    )
    return f'<div class="bot-block"><strong>{heading}</strong><br>{buttons}</div>'

def render_all_phase_groups():
    html = ""
    phases = ["problem_framing", "implementation", "evaluation", "conclusion"]
    for phase in phases:
        methods = top_methods_by_phase(phase)
        if not methods:
            continue
        title_buttons = "".join(
            [f'<button class="method-button" onclick="sendMethod(\'{m["title"]}\')">{m["title"]}</button>' for m in methods]
        )
        html += f'<div class="bot-block"><strong>Top methods in {phase.replace("_", " ").capitalize()} phase:</strong><br>{title_buttons}</div>'
    return html

if __name__ == "__main__":
    app.run(debug=True)
