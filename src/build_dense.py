import json
import time
from pathlib import Path
import lancedb
from sentence_transformers import SentenceTransformer

def build_dense_index(
    chunks_path: str = "corpus_v0/chunks.jsonl",
    db_path: str = "indices/lancedb",
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 64
):
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)

    print(f"Reading chunks from {chunks_path}...")
    data = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    texts_to_embed = [item["embed_text"] for item in data]
    print(f"Total chunks to embed: {len(texts_to_embed)}")

    # Compute embeddings
    start_time = time.time()
    print("Generating dense vector embeddings...")
    embeddings = model.encode(
        texts_to_embed,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    print(f"Embedding completed in {time.time() - start_time:.2f}s")

    # Connect to LanceDB and create table
    db = lancedb.connect(db_path)
    
    records = []
    for item, emb in zip(data, embeddings):
        records.append({
            "chunk_id": item["chunk_id"],
            "paper_id": item["paper_id"],
            "title": item["title"],
            "section": item["section"],
            "header": item["header"],
            "text": item["text"],
            "vector": emb.tolist()
        })

    table_name = "arxiv_chunks"
    print(f"Writing to LanceDB table '{table_name}' at {db_path}...")
    tbl = db.create_table(table_name, data=records, mode="overwrite")
    print(f"Dense index built successfully! Table contains {tbl.count_rows()} records.")

if __name__ == "__main__":
    build_dense_index()
