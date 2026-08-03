"""Query the policy collection.

Returns chunks with their source document, which serves two purposes: the
agent cites sources in its answer, and the eval suite scores retrieval
precision/recall against labelled `expected_source_docs` without needing an LLM.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache

import chromadb

from config import CHROMA_DIR, COLLECTION_NAME, RETRIEVAL_TOP_K


@dataclass
class Chunk:
    text: str
    source_doc: str
    doc_title: str
    section: str
    distance: float

    def as_context(self) -> str:
        return f"[source: {self.source_doc} · {self.section}]\n{self.text}"


@lru_cache(maxsize=1)
def _collection():
    if not CHROMA_DIR.exists():
        raise RuntimeError(
            f"No Chroma store at {CHROMA_DIR}. Run `python -m rag.ingest` first."
        )
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(name=COLLECTION_NAME)


def search(query: str, top_k: int | None = None) -> list[Chunk]:
    """Return the top-k most similar policy chunks."""
    k = top_k if top_k is not None else RETRIEVAL_TOP_K
    result = _collection().query(query_texts=[query], n_results=k)

    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]

    return [
        Chunk(
            text=doc,
            source_doc=meta["source_doc"],
            doc_title=meta["doc_title"],
            section=meta["section"],
            distance=dist,
        )
        for doc, meta, dist in zip(docs, metas, dists)
    ]


def format_context(chunks: list[Chunk]) -> str:
    """Render chunks as the text block handed back to the model."""
    if not chunks:
        return "No relevant policy sections found."
    return "\n\n---\n\n".join(c.as_context() for c in chunks)


def main() -> int:
    query = " ".join(sys.argv[1:]) or "housing stipend for a short term assignment"
    print(f"Query: {query!r}\n")
    for i, chunk in enumerate(search(query), 1):
        print(f"{i}. {chunk.source_doc} · {chunk.section}  (distance {chunk.distance:.4f})")
        print(f"   {chunk.text[:180].replace(chr(10), ' ')}...\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
