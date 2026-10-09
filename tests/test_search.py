import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
from src.retriever import PaperRetriever

def test_retriever():
    print("Testing PaperRetriever across all modes...")
    retriever = PaperRetriever()

    query = "How does Corrective Retrieval Augmented Generation handle low-quality retrieved documents?"
    print(f"\nQuery: {query}")
    print("=" * 80)

    for mode in ["bm25", "dense", "hybrid", "reranked"]:
        print(f"\n>>> Running Mode: {mode.upper()} <<<")
        start = time.time()
        results = retriever.search(query, top_k=3, mode=mode)
        latency = (time.time() - start) * 1000

        print(f"Latency: {latency:.1f} ms | Retrieved {len(results)} chunks")
        for i, doc in enumerate(results, 1):
            print(f"  [{i}] Chunk ID: {doc['chunk_id']} (Paper: {doc['paper_id']})")
            print(f"      Score: {doc['score']:.4f} | Source: {doc['retrieval_source']}")
            print(f"      Header: {doc.get('header', 'N/A')}")
            snippet = doc['text'][:120].replace('\n', ' ')
            print(f"      Snippet: {snippet}...\n")

if __name__ == "__main__":
    test_retriever()
