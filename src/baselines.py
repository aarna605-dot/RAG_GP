from typing import Dict, Any, List
from src.retriever import PaperRetriever

class RAGBaselines:
    """
    Official implementations of S1 and S2 setups for the benchmark suite.
    - S1 (Vanilla RAG): Dense top-5 chunks, single pass.
    - S2 (Tuned RAG): Hybrid BM25 + dense search + cross-encoder reranker + context headers.
    """
    def __init__(self, retriever: PaperRetriever = None):
        self.retriever = retriever or PaperRetriever()

    def s1_vanilla_retrieval(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        S1: Standard single-shot dense retrieval.
        Retrieves top-5 chunks using vector embeddings only.
        """
        return self.retriever.search_dense(query, top_k=top_k)

    def s2_tuned_retrieval(self, query: str, top_k: int = 5, candidate_k: int = 25) -> List[Dict[str, Any]]:
        """
        S2: Tuned retrieval.
        Hybrid BM25 + Dense RRF fusion, filtered and scored with Cross-Encoder reranker.
        """
        return self.retriever.search_reranked(query, top_k=top_k, candidate_k=candidate_k)
