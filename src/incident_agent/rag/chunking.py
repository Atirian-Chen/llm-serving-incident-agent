from __future__ import annotations

import re
from pathlib import Path

from ..config import ROOT


class DocumentChunk:
    """A retrievable runbook section with stable identity and source metadata."""

    def __init__(
        self,
        text: str,
        source: str,
        section: str,
        *,
        chunk_id: str | None = None,
        title: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.text = text.strip()
        self.source = source
        self.section = section
        self.chunk_id = chunk_id or f"{source}::{section}"
        self.title = title or section
        self.metadata = dict(metadata or {})
        self.metadata.setdefault("source", source)
        self.metadata.setdefault("title", self.title)
        self.metadata.setdefault("section", section)

    def as_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "title": self.title,
            "section": self.section,
            "metadata": dict(self.metadata),
        }


def load_markdown_chunks(directory: Path | None = None) -> list[DocumentChunk]:
    directory = directory or (ROOT / "knowledge")
    chunks: list[DocumentChunk] = []
    for path in sorted(directory.glob("*.md")):
        current_section = "overview"
        buffer: list[str] = []
        title = path.stem.replace("_", " ").title()
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                title = line.removeprefix("# ").strip()
                continue
            if line.startswith("## "):
                if "\n".join(buffer).strip():
                    chunks.append(
                        DocumentChunk(
                            "\n".join(buffer), path.name, current_section,
                            chunk_id=f"{path.name}::{len(chunks)}::{current_section}",
                            title=title,
                        )
                    )
                buffer = []
                current_section = line.removeprefix("## ").strip()
            else:
                buffer.append(line)
        if "\n".join(buffer).strip():
            chunks.append(
                DocumentChunk(
                    "\n".join(buffer), path.name, current_section,
                    chunk_id=f"{path.name}::{len(chunks)}::{current_section}",
                    title=title,
                )
            )
    return chunks
