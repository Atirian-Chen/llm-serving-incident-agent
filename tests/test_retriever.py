import pytest

from incident_agent.rag.retriever import KeywordRetriever, load_markdown_chunks


@pytest.mark.asyncio
async def test_retriever_returns_cited_chunks():
    retriever = KeywordRetriever(load_markdown_chunks())
    docs = await retriever.ainvoke("prefix cache 命中率很低")
    assert docs
    assert any("prefix" in doc["source"] for doc in docs)
    assert all(doc["source"] and doc["section"] for doc in docs)

