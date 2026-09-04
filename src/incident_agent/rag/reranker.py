from __future__ import annotations

from typing import Any

from ..config import settings
from .errors import RAGDependencyError, RAGModelError


class CrossEncoderReranker:
    def __init__(
        self,
        *,
        model_name: str | None = None,
        device: str | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name or settings.rag_reranker_model
        self.device = device or settings.rag_device
        if model is None:
            try:
                import torch  # type: ignore
                from sentence_transformers import CrossEncoder  # type: ignore
            except ImportError as exc:
                raise RAGDependencyError(
                    "RAG unavailable: Cross-Encoder reranking requires 'sentence-transformers' and torch; "
                    "install the project's RAG dependencies"
                ) from exc
            if self.device == "cuda" and not torch.cuda.is_available():
                raise RAGModelError(
                    "RAG unavailable: CUDA is required by strict reranker device=cuda but no CUDA device is available; "
                    "install CUDA-enabled torch or fix the NVIDIA driver"
                )
            try:
                self.model = CrossEncoder(self.model_name, device=self.device, max_length=512)
            except Exception as exc:
                raise RAGModelError(
                    f"RAG unavailable: failed to load reranker model {self.model_name!r} on {self.device}; "
                    "check HF_HOME/model cache, GPU memory, and network access"
                ) from exc
        else:
            self.model = model

    def rerank(self, query: str, candidates: list[dict[str, Any]], k: int = 3) -> list[dict[str, Any]]:
        if not candidates:
            return []
        try:
            scores = self.model.predict([(query, str(item.get("text", ""))) for item in candidates])
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
        except Exception as exc:
            message = str(exc)
            if "out of memory" in message.casefold() or "cuda" in message.casefold():
                raise RAGModelError(
                    f"RAG unavailable: reranker GPU execution failed on {self.device} (possible CUDA OOM): {exc}"
                ) from exc
            raise RAGModelError(f"RAG unavailable: Cross-Encoder reranking failed: {exc}") from exc
        ranked = []
        for item, score in zip(candidates, scores):
            result = dict(item)
            result["reranker_score"] = float(score)
            ranked.append(result)
        ranked.sort(key=lambda item: (-float(item["reranker_score"]), str(item["chunk_id"])))
        for rank, item in enumerate(ranked[:k], start=1):
            item["reranker_rank"] = rank
        return ranked[:k]
