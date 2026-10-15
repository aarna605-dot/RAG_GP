
import pytest

from src.verifier import ClaimVerifier


@pytest.fixture
def evidence():
    return [
        {
            "chunk_id": "chunk_1",
            "text": "The method improves retrieval accuracy."
        },
        {
            "chunk_id": "chunk_2",
            "text": "The model uses a transformer architecture."
        },
    ]


def test_valid_citation(evidence):
    verifier = ClaimVerifier()

    result = verifier.verify(
        "The method improves retrieval accuracy. [[chunk_1]]",
        evidence
    )

    assert result["claim_count"] == 1
    assert result["claims"][0]["citation_status"] == "valid"
    assert result["citation_errors"] == 0
    # Citation validity alone does not establish factual support.
    assert result["verified"] is False


def test_invalid_citation(evidence):
    verifier = ClaimVerifier()

    result = verifier.verify(
        "The method is effective. [[unknown_chunk]]",
        evidence
    )

    assert result["claims"][0]["citation_status"] == "invalid"
    assert result["claims"][0]["invalid_citation_ids"] == [
        "unknown_chunk"
    ]
    assert result["verified"] is False


def test_missing_citation(evidence):
    verifier = ClaimVerifier()

    result = verifier.verify(
        "The method improves retrieval accuracy.",
        evidence
    )

    assert result["claims"][0]["citation_status"] == "missing"
    assert result["citation_errors"] == 1


def test_semantic_support_checker(evidence):
    def support_checker(claim, cited_chunks):
        if cited_chunks and "retrieval accuracy" in claim:
            return "supported"
        return "uncertain"

    verifier = ClaimVerifier(support_checker=support_checker)

    result = verifier.verify(
        "The method improves retrieval accuracy. [[chunk_1]]",
        evidence
    )

    assert result["claims"][0]["support_status"] == "supported"
    assert result["verified"] is True


def test_empty_answer(evidence):
    verifier = ClaimVerifier()

    result = verifier.verify("", evidence)

    assert result["claim_count"] == 0
    assert result["verified"] is False


def test_invalid_input_type(evidence):
    verifier = ClaimVerifier()

    with pytest.raises(TypeError):
        verifier.verify(123, evidence)
