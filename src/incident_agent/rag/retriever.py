from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import ROOT, settings


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    source: str
    section: str

    def as_dict(self) -> dict[str, str]:
        return {"text": self.text, "source": self.source, "section": self.section}


def load_markdown_chunks(directory: Path | None = None) -> list[DocumentChunk]:
    directory = directory or (ROOT / "knowledge")
    chunks: list[DocumentChunk] = []
    for path in sorted(directory.glob("*.md")):
        current_section = "overview"
        buffer: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                if buffer:
                    chunks.append(DocumentChunk("\n".join(buffer).strip(), path.name, current_section))
                    buffer = []
                current_section = line.removeprefix("## ").strip()
            elif line.startswith("# "):
                continue
            else:
                buffer.append(line)
        if buffer and "\n".join(buffer).strip():
            chunks.append(DocumentChunk("\n".join(buffer).strip(), path.name, current_section))
    return chunks


def _tokens(text: str) -> set[str]:
    # Keep English metric names and Chinese text usable in the same tiny retriever.
    pieces = re.findall(r"[a-zA-Z][a-zA-Z0-9_./-]*|[\u4e00-\u9fff]", text.casefold())
    return set(pieces)


class KeywordRetriever:
    """Deterministic offline retriever used by tests and local demos."""

    def __init__(self, chunks: list[DocumentChunk] | None = None) -> None:
        self.chunks = chunks if chunks is not None else load_markdown_chunks()
        self._token_sets = [_tokens(chunk.text) | _tokens(chunk.source) for chunk in self.chunks]

    async def ainvoke(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        query_tokens = _tokens(query)
        scored: list[tuple[float, int]] = []
        for index, token_set in enumerate(self._token_sets):
            overlap = len(query_tokens & token_set)
            # Small source-name bonus makes explicit terms like "prefix cache" reliable.
            source_bonus = sum(0.2 for token in query_tokens if token in self.chunks[index].source.casefold())
            scored.append((overlap + source_bonus, index))
        scored.sort(key=lambda item: (-item[0], self.chunks[item[1]].source, self.chunks[item[1]].section))
        selected = [self.chunks[index] for score, index in scored if score > 0][:k]
        if not selected:
            selected = self.chunks[:k]
        return [chunk.as_dict() for chunk in selected]


class ChromaBGERetriever:
    """Optional BGE-small-zh + Chroma implementation for the resume project."""

    def __init__(
        self,
        chunks: list[DocumentChunk] | None = None,
        persist_directory: str | Path | None = None,
        embedding_model: str | None = None,
    ) -> None:
        try:
            import chromadb  # type: ignore
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install llm-serving-incident-agent[rag] for BGE + Chroma") from exc
        self._chunks = chunks if chunks is not None else load_markdown_chunks()
        self._encoder = SentenceTransformer(embedding_model or settings.embedding_model)
        directory = str(persist_directory or settings.chroma_dir)
        client = chromadb.PersistentClient(path=directory)
        self._collection = client.get_or_create_collection("runbooks")
        self._index()

    def _index(self) -> None:  # pragma: no cover - optional dependency
        ids = [f"{chunk.source}:{chunk.section}" for chunk in self._chunks]
        documents = [chunk.text for chunk in self._chunks]
        embeddings = self._encoder.encode(documents, normalize_embeddings=True).tolist()
        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=[{"source": c.source, "section": c.section} for c in self._chunks],
        )

    async def ainvoke(self, query: str, k: int = 4) -> list[dict[str, Any]]:  # pragma: no cover
        embedding = self._encoder.encode([query], normalize_embeddings=True).tolist()
        result = self._collection.query(query_embeddings=embedding, n_results=k)
        docs = result.get("documents", [[]])[0]
        metadata = result.get("metadatas", [[]])[0]
        return [
            {"text": text, "source": meta.get("source", "unknown"), "section": meta.get("section", "")}
            for text, meta in zip(docs, metadata)
        ]


def build_retriever(provider: str | None = None):
    provider = provider or settings.rag_provider
    if provider.lower() in {"chroma", "bge", "dense"}:
        try:
            return ChromaBGERetriever()
        except RuntimeError:
            # Offline startup should still be useful; expose the chosen fallback in README/trace.
            pass
    return KeywordRetriever()

