<h1 align="center">Fuzzy Design Insight Assistant</h1>
<p align="center"><em>A human-centered, conversational assistant for exploring and comparing design-science methods</em></p>

---

## Features

- **Semantic similarity** retrieval (Sentence-Transformers + **FAISS**)
- **Fuzzy phase membership** (graded fit across Problem Framing, Implementation, Evaluation, Conclusion)
- **Soft constraint alignment** (time, resources, group size, facilitation)
- **Transparent UI** - method cards with **Why suggested?**, radial **Similar methods** view, and **Compare** view

---

## 1) Quick Start

```bash
# From repo root
cd backend

# Create & activate virtual environment
python -m venv .venv && .venv\Scripts\activate         # Windows
# or: python3 -m venv .venv && source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

> First run downloads the embedding model and builds the FAISS index. Later runs are instant unless `data.json` changed.

---

## 5) Project Structure (key files)

```
backend/
  app.py                   # Flask server
  templates/index.html     # UI (cards, Why suggested?, radial similar, Compare)
  search_engine.py         # embeddings, FAISS, scoring, MMR, weights
  data.json                # your corpus (you provide)
  data.hash                # content hash for index invalidation
  faiss.index*             # built at first run
  eval.py                  # similarity eval (offline)
  eval_title_fuzzy.py      # title / fuzzy diagnostics
  beir_sanity.py           # encoder sanity checks
```

---

## 2) Add / Prepare the Corpus

Place your corpus at **`backend/data.json`**.
Schema can be found in the data.json

**Index lifecycle:** the app writes a content hash (`data.hash`) and FAISS files.  
Delete `faiss*` and `data.hash` to force a rebuild.

---

## 3) Configuration

- **Embedding model:** set in `backend/search_engine.py` (default `BAAI/bge-small-en-v1.5`).
- **Weights / MMR / Top-K:** tune `WEIGHTS`, diversification params, and `TOP_K` in `backend/search_engine.py`.

---

## 4) Running Evaluations (from `backend/`)

### 4.1 Similarity benchmark (in-domain)

```bash
python eval.py --data data.json --gold benchmark.tsv --k 5
```

**Metrics:** `recall@k_mean`, `map@k_mean`, `ndcg@k_mean`, `coverage_unique`, `coverage_ratio`, `intralist_similarity_mean`.

### 4.2 Title / fuzzy diagnostics

```bash
python eval_title_fuzzy.py --data titles.tsv
```

Reports EM@1 / Hits@K / MRR@K for exact & fuzzy name paths.

### 4.3 Encoder sanity (BEIR-style reference)

```bash
python beir_sanity.py --model all-MiniLM-L6-v2
python beir_sanity.py --model BAAI/bge-small-en-v1.5
```

### 4.4 (Optional) QnA sanity

```bash
python qa_eval.py --qa qa.tsv
```

---


## 7) Appendix Links

- **Code:** <https://github.com/Richa0303/FDSR-Decision-Engine-New.git>  
- **Evaluation sheet:** <https://docs.google.com/spreadsheets/d/1PCFTyQCX95fYUjFVfKlaDS0U6u1zuHnf/edit?usp=sharing&ouid=109155667480468791376&rtpof=true&sd=true>

---

