"""Chunk the policy corpus and load it into a local Chroma collection.

Chunking is section-aware: each `##` heading in a policy document becomes one
chunk, because the corpus was authored so that a section is a self-contained
unit of policy. Sections longer than CHUNK_MAX_CHARS are split on paragraph
boundaries with a small overlap.

Embeddings use Chroma's default local model (all-MiniLM-L6-v2 via onnxruntime).
No API key, no network call after the first model download.

Run:  python -m rag.ingest
"""

from __future__ import annotations

import re
import shutil
import sys

import chromadb

from config import (
    CHROMA_DIR,
    CHUNK_MAX_CHARS,
    CHUNK_OVERLAP_CHARS,
    COLLECTION_NAME,
    CORPUS_DIR,
)


def split_into_sections(markdown: str) -> list[tuple[str, str]]:
    """Split a document on `##` headings into (heading, body) pairs.

    Text before the first `##` (the title and the fabrication notice) is
    attached to the first section rather than dropped, so the document's
    subject is present in the embedding of its opening chunk.
    """
    parts = re.split(r"^## +(.+)$", markdown, flags=re.MULTILINE)
    preamble = parts[0].strip()

    sections: list[tuple[str, str]] = []
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections.append((heading, body))

    if sections and preamble:
        first_heading, first_body = sections[0]
        sections[0] = (first_heading, f"{preamble}\n\n{first_body}")
    elif not sections and preamble:
        sections.append(("Overview", preamble))

    return sections


def split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split on paragraph boundaries, keeping `overlap` chars of tail context."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= max_chars or not current:
            current = candidate
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{para}" if tail else para

    if current:
        chunks.append(current)
    return chunks


def build_chunks() -> list[dict]:
    """Turn the corpus directory into embeddable chunks with source metadata."""
    doc_paths = sorted(CORPUS_DIR.glob("*.md"))
    if not doc_paths:
        raise FileNotFoundError(f"No markdown files found in {CORPUS_DIR}")

    chunks: list[dict] = []
    for path in doc_paths:
        raw = path.read_text(encoding="utf-8")
        title = raw.lstrip().split("\n", 1)[0].lstrip("# ").strip()

        for section_idx, (heading, body) in enumerate(split_into_sections(raw)):
            if not body.strip():
                continue
            pieces = split_long_text(body, CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS)
            for piece_idx, piece in enumerate(pieces):
                # The document title and section heading are prepended to the
                # embedded text. Queries name topics ("housing stipend"), and
                # a mid-document paragraph often never repeats that phrase.
                embed_text = f"{title} — {heading}\n\n{piece}"
                chunks.append(
                    {
                        "id": f"{path.stem}::{section_idx}::{piece_idx}",
                        "text": embed_text,
                        "metadata": {
                            "source_doc": path.name,
                            "doc_title": title,
                            "section": heading,
                        },
                    }
                )
    return chunks


def ingest(reset: bool = True) -> int:
    """Build the collection from scratch. Returns the chunk count."""
    if reset and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    chunks = build_chunks()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    return len(chunks)


def main() -> int:
    print(f"Corpus:  {CORPUS_DIR}")
    print(f"Chroma:  {CHROMA_DIR}")
    print(f"Config:  max_chars={CHUNK_MAX_CHARS} overlap={CHUNK_OVERLAP_CHARS}")

    count = ingest()

    by_doc: dict[str, int] = {}
    for chunk in build_chunks():
        by_doc[chunk["metadata"]["source_doc"]] = (
            by_doc.get(chunk["metadata"]["source_doc"], 0) + 1
        )

    print(f"\nIngested {count} chunks from {len(by_doc)} documents:")
    for doc, n in sorted(by_doc.items()):
        print(f"  {doc:<42} {n:>3} chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
