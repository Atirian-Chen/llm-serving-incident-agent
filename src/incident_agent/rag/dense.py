from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ..config import settings
from .chunking import DocumentChunk
from .errors import RAGDependencyError, RAGModelError


class DenseRetriever:
    """Normalized BGE embeddings indexed in a Chroma cosine collection."""

    def __init__(
        self,
        chunks: list[DocumentChunk],
        *,
        persist_directory: str | Path | None = None,
        embedding_model: str | None = None,
        device: str | None = None,
        encoder: Any | None = None,
        collection: Any | None = None,
    ) -> None:
        self.chunks = chunks
        self.device = device or settings.rag_device
        self.model_name = embedding_model or settings.embedding_model
        if encoder is None:
            try:
                import torch  # type: ignore
                import chromadb  # type: ignore
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as exc:
                raise RAGDependencyError(
                    "RAG unavailable: Dense retrieval requires 'chromadb', 'sentence-transformers', "
                    "and torch; install the project's RAG dependencies"
                ) from exc
            if self.device == "cuda" and not torch.cuda.is_available():
                raise RAGModelError(
                    "RAG unavailable: CUDA is required by strict RAG_DEVICE=cuda but torch.cuda.is_available() is false; "
                    "install a CUDA-enabled torch build and verify the NVIDIA driver"
                )
            try:
                self.encoder = SentenceTransformer(self.model_name, device=self.device)
            except Exception as exc:
                raise RAGModelError(
                    f"RAG unavailable: failed to load embedding model {self.model_name!r} on {self.device}; "
                    "check HF_HOME/model cache and network access"
                ) from exc
        else:
            self.encoder = encoder
        if collection is None:
            try:
                import chromadb  # type: ignore
                client = chromadb.PersistentClient(path=str(persist_directory or settings.chroma_dir))
                self.collection = client.get_or_create_collection(
                    "runbooks_hybrid_v1", metadata={"hnsw:space": "cosine"}
                )
            except ImportError as exc:
                raise RAGDependencyError(
                    "RAG unavailable: missing dependency 'chromadb' for dense retrieval; install chromadb"
                ) from exc
            except Exception as exc:
                raise RAGModelError(
                    f"RAG unavailable: failed to open Chroma cosine collection: {exc}"
                ) from exc
        else:
            self.collection = collection
        self._index()

    def _encode(self, texts: list[str]) -> list[list[float]]:
        try:
            result = self.encoder.encode(texts, normalize_embeddings=True)
            return result.tolist() if hasattr(result, "tolist") else result
        except Exception as exc:
            raise RAGModelError(f"RAG unavailable: embedding execution failed on {self.device}: {exc}") from exc

    def _index(self) -> None:
        if not self.chunks:
            return
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in self.chunks],
            documents=[chunk.text for chunk in self.chunks],
            embeddings=self._encode([chunk.text for chunk in self.chunks]),
            metadatas=[chunk.metadata for chunk in self.chunks],
        )

    def _retrieve_sync(self, query: str, k: int) -> list[dict[str, Any]]:
        embedding = self._encode([query])
        result = self.collection.query(query_embeddings=embedding, n_results=k, include=["documents", "metadatas", "distances"])
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metadata = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        output: list[dict[str, Any]] = []
        for rank, (chunk_id, text, meta, distance) in enumerate(zip(ids, docs, metadata, distances), start=1):
            # With normalized vectors and Chroma's cosine space, similarity is 1 - distance.
            dense_score = 1.0 - float(distance)
            item = dict(meta or {})
            item.update({"chunk_id": chunk_id, "text": text, "dense_score": dense_score, "dense_rank": rank})
            item.setdefault("source", "unknown")
            item.setdefault("title", item.get("section", ""))
            item.setdefault("section", item.get("title", ""))
            item.setdefault("metadata", dict(meta or {}))
            output.append(item)
        return output

    async def retrieve(self, query: str, k: int = 20) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._retrieve_sync, query, k)
