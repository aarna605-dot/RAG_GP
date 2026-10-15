"""R4 worker: retrieve, inspect, and reflect on evidence."""

import json
from typing import Any, Dict, List, Optional

from src.retriever import PaperRetriever
from src.router.schemas import SubQuestion, Usage
from src.router.llm import LLMClient
from src.workers.context import build_context


class RetrievalWorker:
    def __init__(
        self,
        retriever: PaperRetriever,
        top_k: int = 5,
        context_budget: int = 2000,
        search_mode: str = "reranked",
        llm: Optional[LLMClient] = None,
        max_steps: int = 8,
    ):
        self.retriever = retriever
        self.top_k = top_k
        self.context_budget = context_budget
        self.search_mode = search_mode
        self.llm = llm
        self.max_steps = max(1, min(max_steps, 8))

    def run(
        self,
        sub_question: SubQuestion,
        resolved_question: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve evidence and optionally reflect on whether more is needed."""
        question = resolved_question or sub_question.text
        current_query = question

        all_chunks: List[Dict[str, Any]] = []
        seen_ids = set()
        seen_text = set()
        trace = []
        usage = Usage()

        for step in range(1, self.max_steps + 1):
            chunks = self.retriever.search(
                query=current_query,
                top_k=self.top_k,
                mode=self.search_mode,
            )

            new_chunks = []
            for chunk in chunks:
                text = str(chunk.get("text", "")).strip()
                chunk_id = chunk.get("chunk_id")

                if not text:
                    continue

                normalized_text = text.lower()
                if chunk_id is not None and chunk_id in seen_ids:
                    continue
                if normalized_text in seen_text:
                    continue

                new_chunks.append(chunk)
                seen_text.add(normalized_text)
                if chunk_id is not None:
                    seen_ids.add(chunk_id)

            all_chunks.extend(new_chunks)

            context_result = build_context(
                all_chunks,
                max_tokens=self.context_budget,
            )

            trace.append({
                "step": step,
                "query": current_query,
                "new_chunks": len(new_chunks),
                "total_chunks": context_result["chunk_count"],
            })

            # Without an LLM, preserve retrieval-only behavior.
            if self.llm is None:
                break

            if step == self.max_steps:
                break

            system = (
                "You are a research retrieval assistant. Inspect the question "
                "and evidence. Decide whether another search could find useful "
                "missing evidence. Reply with JSON only, using this schema: "
                '{"continue_search": true or false, "query": "next search query"}. '
                "Do not invent evidence. If the evidence is sufficient, set "
                "continue_search to false."
            )

            user = (
                f"Question: {question}\n\n"
                f"Current search query: {current_query}\n\n"
                f"Retrieved evidence:\n{context_result['context']}\n\n"
                "Is another search needed?"
            )

            response = self.llm.complete(system, user, max_tokens=150)
            usage.add(response.usage)

            try:
                decision = json.loads(response.text)
                continue_search = decision.get("continue_search") is True
                next_query = str(decision.get("query", "")).strip()
            except (json.JSONDecodeError, AttributeError):
                # Stop safely if the model returns malformed JSON.
                break

            if not continue_search or not next_query:
                break

            current_query = next_query

        final_context = build_context(
            all_chunks,
            max_tokens=self.context_budget,
        )

        return {
            "sub_question_id": sub_question.id,
            "question": question,
            "chunks": final_context["chunks"],
            "context": final_context["context"],
            "estimated_tokens": final_context["estimated_tokens"],
            "chunk_count": final_context["chunk_count"],
            "retrieval_mode": self.search_mode,
            "steps": len(trace),
            "trace": trace,
            "usage": usage.to_dict(),
        }
