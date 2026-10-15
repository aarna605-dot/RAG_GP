
"""R6: Claim-level citation verification for the Agentic RAG pipeline.

Checks citation existence against retrieved evidence.
Semantic support checking is optional and must be supplied separately.
"""

import re
from typing import Any, Callable, Dict, List, Optional, Set


class ClaimVerifier:
    """Validate citations and optionally check claim-evidence support."""

    # Citation format: [[chunk_id]]
    CITATION_PATTERN = re.compile(r"\[\[([^\[\]]+)\]\]")

    def __init__(
        self,
        support_checker: Optional[Callable[[str, List[Dict[str, Any]]], str]] = None,
    ):
        """
        Args:
            support_checker:
                Optional function receiving (claim, cited_chunks).
                It must return one of:
                "supported", "unsupported", or "uncertain".
                Without this function, semantic support is "uncertain".
        """
        self.support_checker = support_checker

    @staticmethod
    def _get_chunk_id(chunk: Dict[str, Any]) -> Optional[str]:
        """Return a normalized chunk ID, if present."""
        chunk_id = chunk.get("chunk_id")
        if chunk_id is None:
            return None
        return str(chunk_id)

    def verify(
        self,
        answer: str,
        evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Verify citations in an answer against retrieved evidence.

        Evidence items should contain 'chunk_id' and, ideally, 'text'.

        Claims are provisionally split by sentence-ending punctuation.
        A sentence without a citation is reported as uncited.
        """
        if not isinstance(answer, str):
            raise TypeError("answer must be a string")

        if not isinstance(evidence, list):
            raise TypeError("evidence must be a list")

        evidence_by_id: Dict[str, Dict[str, Any]] = {}
        for chunk in evidence:
            if not isinstance(chunk, dict):
                continue
            chunk_id = self._get_chunk_id(chunk)
            if chunk_id is not None:
                evidence_by_id[chunk_id] = chunk

        # Split answer into non-empty sentences.
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer.strip())
            if sentence.strip()
        ]

        results = []
        for index, sentence in enumerate(sentences, start=1):
            citation_ids = [
                citation.strip()
                for citation in self.CITATION_PATTERN.findall(sentence)
                if citation.strip()
            ]

            # Remove citation markers from the claim text.
            claim = self.CITATION_PATTERN.sub("", sentence).strip()
            valid_ids = list(dict.fromkeys(
                cid for cid in citation_ids if cid in evidence_by_id
            ))
            invalid_ids = list(dict.fromkeys(
                cid for cid in citation_ids if cid not in evidence_by_id
            ))

            cited_chunks = [evidence_by_id[cid] for cid in valid_ids]

            if not citation_ids:
                citation_status = "missing"
                support_status = "uncertain"
            elif invalid_ids:
                citation_status = "invalid"
                support_status = "uncertain"
            else:
                citation_status = "valid"
                support_status = "uncertain"

                if self.support_checker is not None:
                    try:
                        verdict = self.support_checker(claim, cited_chunks)
                        if verdict in {"supported", "unsupported", "uncertain"}:
                            support_status = verdict
                    except Exception:
                        # A failed semantic checker must not imply support.
                        support_status = "uncertain"

            results.append({
                "claim_id": index,
                "claim": claim,
                "citation_ids": citation_ids,
                "valid_citation_ids": valid_ids,
                "invalid_citation_ids": invalid_ids,
                "citation_status": citation_status,
                "support_status": support_status,
            })

        citation_errors = sum(
            result["citation_status"] in {"missing", "invalid"}
            for result in results
        )
        unsupported_claims = sum(
            result["support_status"] == "unsupported"
            for result in results
        )
        uncertain_claims = sum(
            result["support_status"] == "uncertain"
            for result in results
        )

        return {
            "verified": (
                bool(results)
                and citation_errors == 0
                and unsupported_claims == 0
                and uncertain_claims == 0
            ),
            "claim_count": len(results),
            "citation_errors": citation_errors,
            "unsupported_claims": unsupported_claims,
            "uncertain_claims": uncertain_claims,
            "claims": results,
        }
