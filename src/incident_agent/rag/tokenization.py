from __future__ import annotations

import re
from collections.abc import Iterable

from .errors import RAGDependencyError

_ASCII_TOKEN = re.compile(r"[a-zA-Z][a-zA-Z0-9_./:-]*|\d+(?:\.\d+)?(?:%|ms|s|gb|mb|m|k)?", re.I)
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_PHRASES = (
    "cuda out of memory",
    "out of memory",
    "gpu memory utilization",
    "scheduler queue",
    "prefix cache",
    "kv cache",
    "continuous batching",
    "pagedattention",
    "tensor parallel",
    "request queue",
    "服务 5xx",
    "请求排队",
    "显存不足",
    "通信异常",
)


def _jieba_tokens(text: str) -> list[str]:
    try:
        import jieba  # type: ignore
    except ImportError as exc:
        raise RAGDependencyError(
            "RAG unavailable: missing dependency 'jieba' for Chinese BM25 tokenization; "
            "install it with pip install jieba"
        ) from exc
    tokens: list[str] = []
    for run in _CJK_RUN.findall(text):
        tokens.extend(token.casefold() for token in jieba.lcut(run, cut_all=False) if token.strip())
    return tokens


def tokenize(text: str, *, require_jieba: bool = True) -> list[str]:
    """Tokenize mixed Chinese/English serving terminology without losing identifiers."""
    normalized = text.casefold()
    tokens: list[str] = [match.group(0) for match in _ASCII_TOKEN.finditer(normalized)]
    # Production BM25 requires jieba. The explicit keyword baseline can be
    # exercised without optional RAG dependencies, so it deliberately keeps
    # ASCII/phrase tokens only when ``require_jieba=False``.
    if require_jieba:
        tokens.extend(_jieba_tokens(normalized))
    # Phrase tokens make incident queries robust while keeping component terms.
    for phrase in _PHRASES:
        if phrase in normalized:
            tokens.append(phrase)
    # Preserve exact metric names and underscore variables even if a segmenter changes.
    tokens.extend(match.group(0) for match in re.finditer(r"\b[a-z][a-z0-9_]*\b", normalized))
    return list(dict.fromkeys(token for token in tokens if token.strip()))


def token_sets(texts: Iterable[str], *, require_jieba: bool = True) -> list[list[str]]:
    return [tokenize(text, require_jieba=require_jieba) for text in texts]
