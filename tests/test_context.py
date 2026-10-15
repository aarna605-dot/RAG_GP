
from src.workers.context import estimate_tokens, build_context


def test_estimate_tokens():
    assert estimate_tokens("hello") == 2


def test_deduplicates_chunks():
    chunks = [
        {"chunk_id": "1", "text": "Same paper text"},
        {"chunk_id": "1", "text": "Same paper text"},
        {"chunk_id": "2", "text": "Different paper text"},
    ]

    result = build_context(chunks)

    assert result["chunk_count"] == 2


def test_respects_token_budget():
    chunks = [
        {"chunk_id": "1", "text": "a" * 40},
        {"chunk_id": "2", "text": "b" * 40},
    ]

    result = build_context(chunks, max_tokens=10)

    assert result["estimated_tokens"] <= 10
    assert result["chunk_count"] == 1
