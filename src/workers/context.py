
"""Context management for R4 workers."""

from typing import Any, Dict, List


def estimate_tokens(text: str) -> int:
    """Rough estimate: approximately 4 characters per token."""
    return max(1, (len(text) + 3) // 4)


def build_context(
    chunks: List[Dict[str, Any]],
    max_tokens: int = 4000,
) -> Dict[str, Any]:
    """
    Deduplicate retrieved chunks and fit them within a token budget.

    Each chunk is expected to contain a 'text' field.
    """
    selected = []
    seen_ids = set()
    seen_text = set()
    used_tokens = 0

    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()

        if not text:
            continue

        chunk_id = chunk.get("chunk_id")
        normalized_text = text.lower()

        if chunk_id is not None and chunk_id in seen_ids:
            continue

        if normalized_text in seen_text:
            continue

        cost = estimate_tokens(text)

        if used_tokens + cost > max_tokens:
            continue

        selected.append(chunk)
        used_tokens += cost
        seen_text.add(normalized_text)

        if chunk_id is not None:
            seen_ids.add(chunk_id)

    context_text = "\n\n".join(
        f"[Source {i + 1}]\n{chunk['text']}"
        for i, chunk in enumerate(selected)
    )

    return {
        "chunks": selected,
        "context": context_text,
        "estimated_tokens": used_tokens,
        "chunk_count": len(selected),
    }
