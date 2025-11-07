import json
from search_engine import infer_phase_from_query  # assumes you already wrote this

with open("test_queries.json") as f:
    test_cases = json.load(f)

correct = 0
for case in test_cases:
    query = case["query"]
    expected = case["expected_phase"]
    predicted = infer_phase_from_query(query)
    print(f"Q: {query}\n → Expected: {expected}, Predicted: {predicted}\n")
    if predicted == expected:
        correct += 1

accuracy = correct / len(test_cases)
print(f"✅ Accuracy: {accuracy:.2%} ({correct}/{len(test_cases)})")
