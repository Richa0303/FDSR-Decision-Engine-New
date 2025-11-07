# tools/beir_to_datajson.py
import json, os, gzip

def read_jsonl(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def convert(corpus_path, out_json):
    data = []
    for doc in read_jsonl(corpus_path):
        # BEIR corpus fields are usually: {"_id": "...", "title": "...", "text": "...", "metadata": {...}}
        did   = doc.get("_id") or doc.get("doc_id")
        title = doc.get("title") or ""
        text  = doc.get("text") or ""
        data.append({
            "id": did,                        # keep BEIR id!
            "title": title,
            "short_description": text[:220],  # light mapping
            "long_description": text,
            "keywords": [],                   # empty; fine
            "phase": "",                      # not used for BEIR
            "phase_membership": {},           # not used for BEIR
        })
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"Wrote {len(data)} docs -> {out_json}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="path to BEIR corpus.jsonl (.gz ok)")
    ap.add_argument("--out", required=True, help="output data.json path")
    args = ap.parse_args()
    convert(args.corpus, args.out)
