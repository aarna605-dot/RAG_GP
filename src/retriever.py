import json
from pathlib import Path
from typing import List, Dict, Any, Literal
import lancedb
import bm25s
from sentence_transformers import SentenceTransformer, CrossEncoder

class PaperRetriever:
    """
    Core Retrieval Engine for Role R2.
    Supports:
      - BM25 (Sparse lexical)
      - Dense (Semantic embeddings via LanceDB)
      - Hybrid RRF (Reciprocal Rank Fusion)
      - Cross-Encoder Reranking (S2 Tuned RAG)
    """
    def __init__(
        self,
        dense_db_path: str = "indices/lancedb",
        bm25_index_dir: str = "indices/bm25",
        dense_model_name: str = "BAAI/bge-small-en-v1.5",
        reranker_model_name: str = "BAAI/bge-reranker-base",
    ):
        print("Initializing PaperRetriever...")
        
        # 1. Connect to LanceDB
        self.db = lancedb.connect(dense_db_path)
        self.dense_table = self.db.open_table("arxiv_chunks")
        print(f"Loaded LanceDB table 'arxiv_chunks' with {self.dense_table.count_rows()} rows.")

        # 2. Dense embedding model
        print(f"Loading dense embedding model: {dense_model_name}...")
        self.dense_model = SentenceTransformer(dense_model_name)

        # 3. BM25 Sparse Index
        print(f"Loading BM25 index from {bm25_index_dir}...")
        self.bm25_retriever = bm25s.BM25.load(bm25_index_dir, load_corpus=False)
        with open(Path(bm25_index_dir) / "metadata.json", "r", encoding="utf-8") as f:
            self.bm25_metadata = json.load(f)
        print(f"Loaded BM25 index with {len(self.bm25_metadata)} metadata entries.")

        # 4. Reranker Cross-Encoder model
        self.reranker_model_name = reranker_model_name
        self._reranker = None  # Lazy load to optimize startup time when only testing sparse/dense

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            print(f"Loading Cross-Encoder reranker: {self.reranker_model_name}...")
            self._reranker = CrossEncoder(self.reranker_model_name)
        return self._reranker

    def search_bm25(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Lexical search via bm25s."""
        query_tokens = bm25s.tokenize([query], stopwords="en")
        results, scores = self.bm25_retriever.retrieve(query_tokens, k=top_k)

        output = []
        for doc_idx, score in zip(results[0], scores[0]):
            meta = self.bm25_metadata[doc_idx].copy()
            meta["score"] = float(score)
            meta["retrieval_source"] = "bm25"
            output.append(meta)
        return output

    def search_dense(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Dense semantic search via LanceDB."""
        query_vector = self.dense_model.encode(query, normalize_embeddings=True)
        results = (
            self.dense_table.search(query_vector.tolist())
            .limit(top_k)
            .to_list()
        )
        output = []
        for r in results:
            item = dict(r)
            item.pop("vector", None)
            item["score"] = float(1.0 - item.get("_distance", 0.0))
            item["retrieval_source"] = "dense"
            output.append(item)
        return output

    def search_hybrid(self, query: str, top_k: int = 20, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion (RRF) combining BM25 and Dense."""
        dense_results = self.search_dense(query, top_k=top_k)
        bm25_results = self.search_bm25(query, top_k=top_k)

        fused_scores = {}
        docs_by_id = {}

        for rank, doc in enumerate(dense_results):
            cid = doc["chunk_id"]
            docs_by_id[cid] = doc
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, doc in enumerate(bm25_results):
            cid = doc["chunk_id"]
            if cid not in docs_by_id:
                docs_by_id[cid] = doc
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))

        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

        output = []
        for cid in sorted_ids[:top_k]:
            doc = docs_by_id[cid].copy()
            doc["score"] = float(fused_scores[cid])
            doc["retrieval_source"] = "hybrid_rrf"
            output.append(doc)
        return output

    def search_reranked(self, query: str, top_k: int = 5, candidate_k: int = 25) -> List[Dict[str, Any]]:
        """S2 Tuned RAG: Candidate generation via Hybrid RRF + Cross-Encoder reranking."""
        candidates = self.search_hybrid(query, top_k=candidate_k)
        if not candidates:
            return []

        # Form (query, document) pairs using contextual header + body
        pairs = [[query, f"{c.get('header', '')}\n{c['text']}"] for c in candidates]
        rerank_scores = self.reranker.predict(pairs)

        for c, score in zip(candidates, rerank_scores):
            c["score"] = float(score)
            c["retrieval_source"] = "reranked"

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: Literal["dense", "bm25", "hybrid", "reranked"] = "reranked"
    ) -> List[Dict[str, Any]]:
        """
        Unified search API for downstream modules (R4, R5, R7, R8).
        """
        if mode == "dense":
            return self.search_dense(query, top_k=top_k)
        elif mode == "bm25":
            return self.search_bm25(query, top_k=top_k)
        elif mode == "hybrid":
            return self.search_hybrid(query, top_k=top_k)
        elif mode == "reranked":
            return self.search_reranked(query, top_k=top_k)
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from 'dense', 'bm25', 'hybrid', 'reranked'.")
