import json
import time
from pathlib import Path
import bm25s

def build_bm25_index(
    chunks_path: str = "corpus_v0/chunks.jsonl",
    index_dir: str = "indices/bm25"
):
    print(f"Loading chunks from {chunks_path}...")
    corpus_texts = []
    chunk_metadata = []

    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            corpus_texts.append(item["embed_text"])
            chunk_metadata.append({
                "chunk_id": item["chunk_id"],
                "paper_id": item["paper_id"],
                "title": item["title"],
                "section": item["section"],
                "header": item["header"],
                "text": item["text"]
            })

    total = len(corpus_texts)
    print(f"Total chunks loaded: {total}")

    # 1. Tokenize the corpus
    start = time.time()
    print("Tokenizing corpus with bm25s...")
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en")
    print(f"Tokenized in {time.time() - start:.2f}s")

    # 2. Build BM25 index
    print("Indexing with BM25...")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    # 3. Save to disk
    out_dir = Path(index_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    retriever.save(str(out_dir))

    # Save metadata mapping
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunk_metadata, f)

    print(f"BM25 index and metadata successfully saved in {out_dir}/")

if __name__ == "__main__":
    build_bm25_index()
