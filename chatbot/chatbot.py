import json

# Load method data from JSON
def load_methods():
    with open("data.json", "r", encoding="utf-8") as file:
        return json.load(file)

# Search method by keyword in title or description
def search_methods(query):
    query = query.lower()
    methods = load_methods()
    results = []
    for method in methods:
        text = method["title"].lower() + " " + method.get("short_description", "").lower()
        if query in text:
            results.append(method)
    return results

# List methods grouped by phase membership
def list_methods_by_phase(methods):
    categories = {
        "problem_framing": [],
        "implementation": [],
        "evaluation": [],
        "conclusion": []
    }
    for m in methods:
        for phase in categories:
            if m.get("phase_membership", {}).get(phase, 0) >= 0.7:
                categories[phase].append(m["title"])

    print("\n📂 Explore Methods by Category:")
    for cat, names in categories.items():
        emoji = {
            "problem_framing": "📌",
            "implementation": "🔧",
            "evaluation": "🧪",
            "conclusion": "✅"
        }[cat]
        print(f"\n{emoji} {cat.replace('_', ' ').title()}:")
        for n in names:
            print(f"   ➤ {n}")

# Run chatbot CLI
methods = load_methods()
last_method = None

print("🤖 Hello! I can help you explore transdisciplinary methods from the td-net toolbox.")
print("💡 You can ask about a method (e.g., 'delphi') or type 'list' to explore by category.")

while True:
    user_input = input("🧑 You: ").strip().lower()
    if user_input in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    if last_method and user_input == 'steps':
        steps = last_method.get('steps', [])
        if steps:
            print("📝 Steps:")
            for s in steps:
                print(f"▶️ {s}")
        else:
            print("❌ No steps found.")
        continue

    if last_method and user_input == 'source':
        print(f"🔗 Source: {last_method.get('source', 'No source available')}")
        continue

    if user_input == "list":
        list_methods_by_phase(methods)
        continue

    matches = search_methods(user_input)
    if not matches:
        print("❌ Hmm, I couldn't find a matching method. Try a different keyword.")
        last_method = None
    else:
        last_method = matches[0]
        print(f"\n✅ {last_method['title']}")
        print(f"🔍 {last_method.get('short_description', 'No description available')}")
        print("💬 Want to see the steps or source? (Type 'steps' or 'source')")
