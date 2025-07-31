import json
from sentence_transformers import SentenceTransformer

# Load your method data
with open("data.json", "r", encoding="utf-8") as f:
    methods = json.load(f)

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Add embeddings for each method based on its long_description
for method in methods:
    description = method.get("long_description", "")
    embedding = model.encode(description).tolist()
    method["embedding"] = embedding

# Save updated methods with embeddings
with open("data.json", "w", encoding="utf-8") as f:
    json.dump(methods, f, indent=2)
