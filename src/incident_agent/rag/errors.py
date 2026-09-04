from __future__ import annotations


class RAGError(RuntimeError):
    """Base class for actionable strict-RAG failures."""


class RAGDependencyError(RAGError):
    """A required retrieval dependency is not importable."""


class RAGModelError(RAGError):
    """An embedding or reranker model could not be loaded or executed."""


class RAGConfigurationError(RAGError):
    """RAG configuration is invalid for strict mode."""
