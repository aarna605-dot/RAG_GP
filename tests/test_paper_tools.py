# In tests/test_paper_tools.py
import csv
import json

import pytest
from src.mcp_tools import paper_tools as tools


@pytest.fixture
def sample_corpus(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus_v0"
    corpus.mkdir()

    rows = [
        {
            "chunk_id": "P001_0000",
            "paper_id": "P001",
            "title": "RAG Evaluation",
            "section": "Abstract",
            "position": 0,
            "header": "RAG Evaluation > Abstract",
            "text": "RAG combines retrieval with generation.",
            "embed_text": "RAG Evaluation Abstract: RAG combines retrieval with generation.",
        },
        {
            "chunk_id": "P001_0001",
            "paper_id": "P001",
            "title": "RAG Evaluation",
            "section": "Conclusion",
            "position": 1,
            "header": "RAG Evaluation > Conclusion",
            "text": "Verification detects unsupported claims.",
            "embed_text": "RAG Evaluation Conclusion: Verification detects unsupported claims.",
        },
    ]

    with (corpus / "chunks.jsonl").open(
        "w", encoding="utf-8"
    ) as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    csv_path = tmp_path / "papers.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["arxiv_id", "title", "summary"],
        )
        writer.writeheader()
        writer.writerow({
            "arxiv_id": "P001",
            "title": "RAG Evaluation",
            "summary": "A paper about retrieval and generation.",
        })

    monkeypatch.setattr(tools, "CORPUS_PATH", corpus)
    monkeypatch.setattr(tools, "PAPERS_CSV", csv_path)
    tools.load_chunks.cache_clear()

    yield

    tools.load_chunks.cache_clear()


def test_search(sample_corpus):
    assert tools.search("retrieval")[0]["paper_id"] == "P001"


def test_read_full_paper(sample_corpus):
    result = tools.read("P001")
    assert "retrieval with generation" in result
    assert "unsupported claims" in result


def test_read_section(sample_corpus):
    result = tools.read("P001", "Conclusion")
    assert "unsupported claims" in result
    assert "retrieval with generation" not in result


def test_keyword_grep(sample_corpus):
    results = tools.keyword_grep("unsupported claims")
    assert results[0]["paper_id"] == "P001"


def test_unknown_paper(sample_corpus):
    with pytest.raises(ValueError):
        tools.read("UNKNOWN")