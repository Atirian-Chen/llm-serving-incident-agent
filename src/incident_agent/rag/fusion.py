from __future__ import annotations

from typing import Any


class RRFFusion:
    def __init__(self, rrf_k: int = 60) -> None:
        self.rrf_k = rrf_k

    def fuse(
        self,
        bm25_results: list[dict[str, Any]],
        dense_results: list[dict[str, Any]],
        k: int = 30,
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for result in bm25_results:
            chunk_id = str(result["chunk_id"])
            item = merged.setdefault(chunk_id, dict(result))
            item.update({key: value for key, value in result.items() if key not in item or item[key] is None})
            item["bm25_rank"] = int(result.get("bm25_rank", 0))
            item["rrf_score"] = item.get("rrf_score", 0.0) + 1.0 / (self.rrf_k + item["bm25_rank"])
        for result in dense_results:
            chunk_id = str(result["chunk_id"])
            item = merged.setdefault(chunk_id, dict(result))
            item.update({key: value for key, value in result.items() if key not in item or item[key] is None})
            item["dense_rank"] = int(result.get("dense_rank", 0))
            item["rrf_score"] = item.get("rrf_score", 0.0) + 1.0 / (self.rrf_k + item["dense_rank"])
        ranked = sorted(merged.values(), key=lambda item: (-float(item.get("rrf_score", 0.0)), str(item["chunk_id"])))
        for rank, item in enumerate(ranked[:k], start=1):
            # Keep a stable candidate contract even when a chunk is returned
            # by only one first-stage retriever.
            item.setdefault("bm25_score", None)
            item.setdefault("bm25_rank", None)
            item.setdefault("dense_score", None)
            item.setdefault("dense_rank", None)
            item.setdefault("rrf_score", 0.0)
            item["rrf_rank"] = rank
        return ranked[:k]
