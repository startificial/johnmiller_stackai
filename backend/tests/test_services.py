"""Tests for service layer."""

import pytest

from app.services.rag_service import TextExtractor


@pytest.mark.asyncio
async def test_rag_service_process_query():
    """Test RAG service query processing."""
    service = TextExtractor()
    result = await service.process_query("test query")
    assert "query" in result
    assert "response" in result

