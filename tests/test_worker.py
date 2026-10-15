
from src.router.schemas import SubQuestion
from src.workers.worker import RetrievalWorker


class FakeRetriever:
    def search(self, query, top_k=5, mode="reranked"):
        return [
            {
                "chunk_id": "chunk-1",
                "text": f"Evidence related to {query}",
                "paper_id": "paper-1",
            }
        ]


def test_worker_returns_evidence():
    worker = RetrievalWorker(
        retriever=FakeRetriever(),
        top_k=3,
        context_budget=100,
    )

    sub_question = SubQuestion(
        id="q1",
        text="How does retrieval augmentation work?",
    )

    result = worker.run(sub_question)

    assert result["sub_question_id"] == "q1"
    assert result["chunk_count"] == 1
    assert "retrieval augmentation" in result["context"]
    assert result["retrieval_mode"] == "reranked"


def test_worker_uses_resolved_question():
    worker = RetrievalWorker(retriever=FakeRetriever())

    sub_question = SubQuestion(
        id="q2",
        text="What are its limitations? {q1}",
        depends_on=["q1"],
    )

    result = worker.run(
        sub_question,
        resolved_question="What are its limitations? Method A",
    )

    assert "Method A" in result["question"]

from src.router.llm import MockLLM


def test_worker_refines_search_query():
    class TrackingRetriever(FakeRetriever):
        def __init__(self):
            self.queries = []

        def search(self, query, top_k=5, mode="reranked"):
            self.queries.append(query)
            return [{
                "chunk_id": f"chunk-{len(self.queries)}",
                "text": f"Evidence for {query}",
            }]

    retriever = TrackingRetriever()
    llm = MockLLM([
        '{"continue_search": true, "query": "improved search"}',
        '{"continue_search": false, "query": ""}',
    ])

    worker = RetrievalWorker(retriever=retriever, llm=llm)
    result = worker.run(SubQuestion(id="q1", text="original question"))

    assert retriever.queries == ["original question", "improved search"]
    assert result["steps"] == 2
    assert result["chunk_count"] == 2


def test_worker_never_exceeds_eight_steps():
    class TrackingRetriever(FakeRetriever):
        def __init__(self):
            self.queries = []

        def search(self, query, top_k=5, mode="reranked"):
            self.queries.append(query)
            return [{
                "chunk_id": f"chunk-{len(self.queries)}",
                "text": f"Evidence for {query}",
            }]

    retriever = TrackingRetriever()
    llm = MockLLM([
        '{"continue_search": true, "query": "search again"}',
    ])

    worker = RetrievalWorker(
        retriever=retriever,
        llm=llm,
        max_steps=20,
    )
    result = worker.run(SubQuestion(id="q1", text="original question"))

    assert len(retriever.queries) == 8
    assert result["steps"] == 8