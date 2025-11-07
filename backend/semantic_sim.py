from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

def load_methods():
    with open("data.json", "r", encoding="utf-8") as f:
        data = json.load(f)


def build_embeddings(methods_data):
    titles = [m["title"] for m in methods_data]
    embeddings = [m["embedding"] for m in methods_data]  # ensure these are 1D vectors (lists of floats)
    return titles, np.array(embeddings)

def get_similar_methods(index, titles, embeddings, top_n=6, return_scores=False):
    query_embedding = embeddings[index].reshape(1, -1)
    scores = cosine_similarity(query_embedding, embeddings)[0]
    
    # Pair titles with similarity scores and exclude the method itself
    sims = [{"title": titles[i], "score": float(scores[i])} for i in range(len(scores)) if i != index]
    
    # Sort and return top N
    top_similar = sorted(sims, key=lambda x: x["score"], reverse=True)[:top_n]
    return top_similar if return_scores else [s["title"] for s in top_similar]
