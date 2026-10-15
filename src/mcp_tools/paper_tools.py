import csv
import json
import re
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = ROOT / "corpus_v0"
PAPERS_CSV = ROOT / "papers.csv"


def normalise_id(value):
    """Make corpus paper_id and CSV arxiv_id comparable."""
    value = str(value or "").strip().lower()
    return re.sub(r"v\d+$", "", value)


def get_field(row, *names):
    """Read a field despite minor differences in column naming."""
    fields = {
        re.sub(r"[^a-z0-9]", "", str(k).lower()): v
        for k, v in row.items()
    }

    for name in names:
        key = re.sub(r"[^a-z0-9]", "", name.lower())
        value = fields.get(key)

        if value is not None and str(value).strip():
            return str(value).strip()

    return ""


def read_json_records(path):
    """Read one JSON chunk per line, or a regular JSON file."""
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if isinstance(record, dict):
                        yield record

    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(data, list):
            yield from (item for item in data if isinstance(item, dict))

        elif isinstance(data, dict):
            # Handle {"chunks": [...]} or a single chunk object.
            for key in ("chunks", "records", "data", "items"):
                if isinstance(data.get(key), list):
                    yield from (
                        item for item in data[key]
                        if isinstance(item, dict)
                    )
                    return

            yield data


def get_corpus_files():
    """Support either a corpus directory or one corpus JSON file."""
    if CORPUS_PATH.is_file():
        return [CORPUS_PATH]

    if CORPUS_PATH.is_dir():
        return sorted(
            path for path in CORPUS_PATH.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".jsonl", ".ndjson", ".json"}
        )

    raise FileNotFoundError(
        f"Cannot find the corpus at {CORPUS_PATH}"
    )


@lru_cache(maxsize=1)
def load_chunks():
    """Load corpus chunks and enrich them with CSV metadata."""
    metadata = {}

    if PAPERS_CSV.is_file():
        with PAPERS_CSV.open(
            "r", encoding="utf-8-sig", newline=""
        ) as f:
            for row in csv.DictReader(f):
                paper_id = get_field(
                    row, "arxiv_id", "paper_id", "id"
                )

                if paper_id:
                    metadata[normalise_id(paper_id)] = row

    chunks = []

    for path in get_corpus_files():
        for row in read_json_records(path):
            paper_id = get_field(
                row, "paper_id", "arxiv_id", "id"
            )
            text = get_field(row, "text", "chunk_text")

            # The actual text field is preferred. embed_text is a fallback.
            if not text:
                text = get_field(row, "embed_text")

            if not paper_id or not text:
                continue

            meta = metadata.get(normalise_id(paper_id), {})

            position = get_field(row, "position")
            try:
                position = int(position)
            except (ValueError, TypeError):
                position = 0

            chunks.append({
                "paper_id": paper_id,
                "chunk_id": get_field(row, "chunk_id"),
                "title": (
                    get_field(meta, "title")
                    or get_field(row, "title")
                    or paper_id
                ),
                "section": get_field(row, "section"),
                "header": get_field(row, "header"),
                "position": position,
                "text": text,
                "summary": get_field(meta, "summary"),
                "published": get_field(meta, "published"),
            })

    if not chunks:
        raise RuntimeError(
            "No corpus chunks loaded. Check corpus_v0's file format "
            "and confirm it contains JSON or JSONL chunk records."
        )

    return tuple(sorted(
        chunks,
        key=lambda c: (
            normalise_id(c["paper_id"]),
            c["position"],
            c["chunk_id"],
        ),
    ))


def make_snippet(text, terms, size=350):
    text = re.sub(r"\s+", " ", text).strip()
    lowered = text.lower()

    positions = [
        lowered.find(term)
        for term in terms
        if term and lowered.find(term) >= 0
    ]

    if not positions:
        return text[:size]

    start = max(0, min(positions) - 80)
    return text[start:start + size]


def search(query: str, limit: int = 5) -> list[dict]:
    """
    Basic lexical search over corpus chunks.
    This is not R2's hybrid BM25+dense retrieval.
    """
    terms = set(re.findall(r"\w+", query.lower()))

    if not terms:
        return []

    results = []

    for chunk in load_chunks():
        content = chunk["text"].lower()
        title = chunk["title"].lower()
        section = chunk["section"].lower()

        score = sum(content.count(term) for term in terms)
        score += 3 * sum(term in title for term in terms)
        score += sum(term in section for term in terms)

        if score:
            results.append({
                "paper_id": chunk["paper_id"],
                "chunk_id": chunk["chunk_id"],
                "title": chunk["title"],
                "section": chunk["section"],
                "score": score,
                "snippet": make_snippet(chunk["text"], terms),
            })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:max(1, min(limit, 50))]


def read(paper_id: str, section: str = "") -> str:
    """Read all chunks of a paper, optionally limited to a section."""
    wanted_id = normalise_id(paper_id)

    matches = [
        chunk for chunk in load_chunks()
        if normalise_id(chunk["paper_id"]) == wanted_id
    ]

    if not matches:
        raise ValueError(f"Paper not found: {paper_id}")

    if section.strip():
        requested_section = section.strip().lower()

        matches = [
            chunk for chunk in matches
            if requested_section in chunk["section"].lower()
            or requested_section in chunk["header"].lower()
        ]

        if not matches:
            raise ValueError(
                f"Section '{section}' not found in paper {paper_id}"
            )

    return "\n\n".join(
        f"[{chunk['title']} | {chunk['section']} | "
        f"chunk={chunk['chunk_id']}]\n{chunk['text']}"
        for chunk in matches
    )


def keyword_grep(
    keyword: str,
    limit: int = 20,
) -> list[dict]:
    """Find literal keyword matches without embeddings or indices."""
    keyword = keyword.strip().lower()

    if not keyword:
        return []

    matches = []

    for chunk in load_chunks():
        if keyword in chunk["text"].lower():
            matches.append({
                "paper_id": chunk["paper_id"],
                "chunk_id": chunk["chunk_id"],
                "title": chunk["title"],
                "section": chunk["section"],
                "match": keyword,
                "snippet": make_snippet(chunk["text"], {keyword}),
            })

            if len(matches) >= max(1, min(limit, 100)):
                break

    return matches