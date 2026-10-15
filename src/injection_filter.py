
"""R6: Prompt injection detection for retrieved RAG evidence.

This module flags suspicious instructions in retrieved text.
It does not modify the original text or guarantee detection of every attack.
"""

import re
from typing import Any, Dict, List


class InjectionFilter:
    """Detect common prompt-injection patterns in retrieved text."""

    PATTERNS = {
        "instruction_override": [
            r"\bignore\s+(all\s+)?previous\s+instructions\b",
            r"\bdisregard\s+(all\s+)?(previous|prior|above)\s+instructions\b",
            r"\bforget\s+(all\s+)?previous\s+instructions\b",
        ],
        "role_manipulation": [
            r"\byou\s+are\s+now\s+(an?\s+)?(?!the\s+author\b)[\w-]+",
            r"\bact\s+as\s+(an?\s+)?(?:unrestricted|unfiltered|different)\b",
            r"\bpretend\s+you\s+have\s+no\s+restrictions\b",
        ],
        "system_prompt_attack": [
            r"\breveal\s+(the\s+)?(system prompt|hidden instructions)\b",
            r"\bshow\s+me\s+the\s+system prompt\b",
            r"\bprint\s+(your\s+)?(system prompt|hidden instructions)\b",
        ],
        "data_exfiltration": [
            r"\breveal\s+(all\s+)?(secrets|passwords|api keys|credentials)\b",
            r"\bexfiltrate\b",
            r"\bsend\s+(the\s+)?(api keys|passwords|credentials)\b",
        ],
        "instruction_to_ignore_safety": [
            r"\bdisable\s+(all\s+)?safety\s+(rules|checks|filters)\b",
            r"\bbypass\s+(all\s+)?(safety|security)\s+(rules|checks|filters)\b",
        ],
        "prompt_injection_markers": [
            r"\bBEGIN\s+INSTRUCTIONS\b",
            r"\bEND\s+INSTRUCTIONS\b",
            r"<\s*/?\s*system\s*>",
            r"\[\s*INST\s*\]",
        ],
    }

    def __init__(self) -> None:
        self._compiled_patterns = {
            category: [
                re.compile(pattern, re.IGNORECASE)
                for pattern in patterns
            ]
            for category, patterns in self.PATTERNS.items()
        }

    def inspect_text(self, text: str) -> Dict[str, Any]:
        """Inspect a single text string and report suspicious patterns."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        findings = []

        for category, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    findings.append({
                        "category": category,
                        "matched_text": match.group(0),
                        "start": match.start(),
                        "end": match.end(),
                    })

        categories = sorted({
            finding["category"] for finding in findings
        })

        return {
            "flagged": bool(findings),
            "risk_level": (
                "high" if len(categories) >= 2
                else "medium" if findings
                else "low"
            ),
            "categories": categories,
            "findings": findings,
            "text": text,
        }

    def inspect_chunks(
        self,
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Inspect retrieved chunks while preserving their original content."""
        if not isinstance(chunks, list):
            raise TypeError("chunks must be a list")

        results = []

        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                results.append({
                    "index": index,
                    "chunk_id": None,
                    "flagged": False,
                    "risk_level": "unknown",
                    "categories": [],
                    "findings": [],
                    "error": "Chunk must be a dictionary.",
                })
                continue

            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text", "")

            if not isinstance(text, str):
                text = str(text) if text is not None else ""

            inspection = self.inspect_text(text)

            results.append({
                "index": index,
                "chunk_id": chunk_id,
                **inspection,
            })

        flagged_count = sum(
            result.get("flagged", False)
            for result in results
        )

        return {
            "total_chunks": len(chunks),
            "flagged_chunks": flagged_count,
            "clean_chunks": len(chunks) - flagged_count,
            "results": results,
        }
