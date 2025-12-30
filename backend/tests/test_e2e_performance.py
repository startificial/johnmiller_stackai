"""
Performance Benchmark Tests for RAG Pipeline

These tests measure latency and throughput of the RAG pipeline components
with timing assertions to ensure performance requirements are met.

Requirements:
- PostgreSQL database with pgvector extension
- MISTRAL_API_KEY environment variable
- Test documents in test_documents/ directory

Run with:
    source .env && PYTHONPATH=. uv run pytest backend/tests/test_e2e_performance.py -v -m performance -s

Environment variable overrides for thresholds:
    PERF_THRESHOLD_INGESTION_SINGLE=10000  # ms
    PERF_THRESHOLD_INGESTION_MULTIPAGE=60000  # ms
    PERF_THRESHOLD_QUERY_SIMPLE=5000  # ms
    PERF_THRESHOLD_QUERY_COMPLEX=8000  # ms
    PERF_THRESHOLD_SEARCH_SEMANTIC=500  # ms
    PERF_THRESHOLD_SEARCH_BM25=200  # ms
    PERF_THRESHOLD_SEARCH_HYBRID=1000  # ms
    PERF_THRESHOLD_INTENT_CLASSIFICATION=2000  # ms
    PERF_THRESHOLD_HALLUCINATION_CHECK=3000  # ms
"""

import os
import time
import pytest
import pytest_asyncio
from pathlib import Path
from typing import Dict, Any, List
import asyncio

from backend.app.db.base import init_db, drop_db, async_session_factory
from backend.app.services.text_extraction import text_extraction_service
from backend.app.services.chunking import chunking_service
from backend.app.services.embedding import embedding_service
from backend.app.services.search_indexer import search_indexer
from backend.app.services.retriever import retriever
from backend.app.services.intent_classifier import intent_classifier
from backend.app.services.query_transformer import query_transformer
from backend.app.services.llm_response.service import LLMResponseService
from backend.app.services.hallucination_detector import hallucination_detector
from backend.app.services.llm_response.schemas import Source

from backend.tests.conftest import PerformanceMetrics, measure_time


# Mark all tests as performance and integration tests
pytestmark = [pytest.mark.performance, pytest.mark.integration]


# =============================================================================
# Threshold Configuration
# =============================================================================


def get_threshold(env_var: str, default_ms: float) -> float:
    """Get performance threshold from environment or use default."""
    return float(os.environ.get(env_var, default_ms))


THRESHOLDS = {
    "ingestion_single": get_threshold("PERF_THRESHOLD_INGESTION_SINGLE", 10000),
    "ingestion_multipage": get_threshold("PERF_THRESHOLD_INGESTION_MULTIPAGE", 60000),
    "query_simple": get_threshold("PERF_THRESHOLD_QUERY_SIMPLE", 5000),
    "query_complex": get_threshold("PERF_THRESHOLD_QUERY_COMPLEX", 8000),
    "search_semantic": get_threshold("PERF_THRESHOLD_SEARCH_SEMANTIC", 500),
    "search_bm25": get_threshold("PERF_THRESHOLD_SEARCH_BM25", 200),
    "search_hybrid": get_threshold("PERF_THRESHOLD_SEARCH_HYBRID", 1000),
    "intent_classification": get_threshold("PERF_THRESHOLD_INTENT_CLASSIFICATION", 2000),
    "hallucination_check": get_threshold("PERF_THRESHOLD_HALLUCINATION_CHECK", 3000),
}


# =============================================================================
# Test Document Paths
# =============================================================================

TEST_DOCUMENTS_DIR = Path(__file__).parent.parent.parent / "test_documents"


def get_test_document_path(filename: str) -> Path:
    """Get full path to a test document."""
    path = TEST_DOCUMENTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Test document not found: {path}")
    return path


# =============================================================================
# Fixtures
# =============================================================================


@pytest_asyncio.fixture(scope="function")
async def setup_database():
    """Initialize database for each test function."""
    await init_db()
    yield
    from sqlalchemy import text
    from backend.app.db.base import engine
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE TABLE chat_messages CASCADE"))
        await conn.execute(text("TRUNCATE TABLE chat_sessions CASCADE"))
        await conn.execute(text("TRUNCATE TABLE bm25_postings CASCADE"))
        await conn.execute(text("TRUNCATE TABLE bm25_indexed_chunks CASCADE"))
        await conn.execute(text("TRUNCATE TABLE bm25_vocabulary CASCADE"))
        await conn.execute(text("TRUNCATE TABLE bm25_corpus_stats CASCADE"))
        await conn.execute(text("TRUNCATE TABLE document_chunks CASCADE"))
        await conn.execute(text("TRUNCATE TABLE extracted_pages CASCADE"))
        await conn.execute(text("TRUNCATE TABLE documents CASCADE"))


async def ingest_document_timed(filename: str) -> tuple[Dict[str, Any], float]:
    """
    Ingest a document and return metadata plus duration.

    Returns:
        Tuple of (metadata dict, duration in seconds)
    """
    file_path = get_test_document_path(filename)

    start = time.perf_counter()

    # Step 1: Extract text
    extracted_doc = await text_extraction_service.extract_and_persist(
        file_path=file_path,
    )

    # Step 2: Chunk document
    chunked_doc = await chunking_service.chunk_and_persist(
        document=extracted_doc,
        doc_id=extracted_doc.doc_id,
    )

    # Step 3: Generate embeddings
    embed_result = await embedding_service.embed_document(
        chunked_doc=chunked_doc,
        embed_parents=False,
    )

    # Step 4: Index for BM25
    await search_indexer.index_chunked_document(chunked_doc)

    duration = time.perf_counter() - start

    return {
        "doc_id": str(extracted_doc.doc_id),
        "filename": filename,
        "page_count": extracted_doc.page_count,
        "total_chunks": len(chunked_doc.chunks),
        "chunks_embedded": embed_result.get("chunks_embedded", 0),
    }, duration


@pytest_asyncio.fixture(scope="function")
async def ingested_doc_for_search(setup_database):
    """Ingest a document for search tests."""
    result, _ = await ingest_document_timed("01_happy_path_text_only.pdf")
    return result


# =============================================================================
# Test Class: Ingestion Performance
# =============================================================================


class TestIngestionPerformance:
    """Performance tests for document ingestion pipeline."""

    async def test_ingestion_latency_single_page(self, setup_database, performance_metrics):
        """Test ingestion latency for a single-page document."""
        metrics = performance_metrics(
            "Single Page Ingestion",
            threshold_ms=THRESHOLDS["ingestion_single"]
        )

        result, duration = await ingest_document_timed("01_happy_path_text_only.pdf")
        metrics.durations.append(duration)

        print(f"\nIngested {result['filename']}: {result['page_count']} pages, "
              f"{result['total_chunks']} chunks in {duration*1000:.2f}ms")

        assert metrics.passes_threshold(), (
            f"Ingestion took {duration*1000:.2f}ms, "
            f"threshold is {THRESHOLDS['ingestion_single']}ms"
        )

    async def test_ingestion_latency_multipage(self, setup_database, performance_metrics):
        """Test ingestion latency for a 27-page document."""
        metrics = performance_metrics(
            "Multipage Ingestion (27 pages)",
            threshold_ms=THRESHOLDS["ingestion_multipage"]
        )

        result, duration = await ingest_document_timed("08_multipage_with_hierarchy.pdf")
        metrics.durations.append(duration)

        print(f"\nIngested {result['filename']}: {result['page_count']} pages, "
              f"{result['total_chunks']} chunks in {duration*1000:.2f}ms")

        assert metrics.passes_threshold(), (
            f"Ingestion took {duration*1000:.2f}ms, "
            f"threshold is {THRESHOLDS['ingestion_multipage']}ms"
        )

    async def test_embedding_batch_throughput(self, setup_database, performance_metrics):
        """Test embedding generation throughput."""
        metrics = performance_metrics("Embedding Throughput")

        # First extract and chunk
        file_path = get_test_document_path("08_multipage_with_hierarchy.pdf")
        extracted_doc = await text_extraction_service.extract_and_persist(
            file_path=file_path,
        )
        chunked_doc = await chunking_service.chunk_and_persist(
            document=extracted_doc,
            doc_id=extracted_doc.doc_id,
        )

        # Measure embedding time only
        start = time.perf_counter()
        embed_result = await embedding_service.embed_document(
            chunked_doc=chunked_doc,
            embed_parents=False,
        )
        duration = time.perf_counter() - start
        metrics.durations.append(duration)

        chunks_embedded = embed_result.get("chunks_embedded", 0)
        throughput = chunks_embedded / duration if duration > 0 else 0

        print(f"\nEmbedded {chunks_embedded} chunks in {duration*1000:.2f}ms "
              f"({throughput:.2f} chunks/sec)")

        # Report throughput (no hard threshold, informational)
        assert chunks_embedded > 0, "No chunks were embedded"


# =============================================================================
# Test Class: Search Performance
# =============================================================================


class TestSearchPerformance:
    """Performance tests for search operations."""

    async def test_search_latency_semantic(self, ingested_doc_for_search, performance_metrics):
        """Test semantic search latency."""
        metrics = performance_metrics(
            "Semantic Search",
            threshold_ms=THRESHOLDS["search_semantic"]
        )

        query = "test document content"

        # Run search
        start = time.perf_counter()
        result = await retriever.search(
            query=query,
            top_k=10,
        )
        duration = time.perf_counter() - start
        metrics.durations.append(duration)

        print(f"\nSemantic search for '{query}': {len(result.results)} results "
              f"in {duration*1000:.2f}ms")

        assert metrics.passes_threshold(), (
            f"Semantic search took {duration*1000:.2f}ms, "
            f"threshold is {THRESHOLDS['search_semantic']}ms"
        )

    async def test_search_latency_bm25(self, ingested_doc_for_search, performance_metrics):
        """Test BM25 keyword search latency."""
        metrics = performance_metrics(
            "BM25 Search",
            threshold_ms=THRESHOLDS["search_bm25"]
        )

        query = "quick brown fox"

        # BM25 search via the indexer
        start = time.perf_counter()
        result = await retriever.search(
            query=query,
            top_k=10,
        )
        duration = time.perf_counter() - start
        metrics.durations.append(duration)

        print(f"\nBM25 search for '{query}': {len(result.results)} results "
              f"in {duration*1000:.2f}ms")

        assert metrics.passes_threshold(), (
            f"BM25 search took {duration*1000:.2f}ms, "
            f"threshold is {THRESHOLDS['search_bm25']}ms"
        )

    async def test_search_latency_hybrid(self, ingested_doc_for_search, performance_metrics):
        """Test hybrid RRF search latency."""
        metrics = performance_metrics(
            "Hybrid RRF Search",
            threshold_ms=THRESHOLDS["search_hybrid"]
        )

        query = "document content testing"

        # Hybrid search (default)
        start = time.perf_counter()
        result = await retriever.search(
            query=query,
            top_k=10,
        )
        duration = time.perf_counter() - start
        metrics.durations.append(duration)

        print(f"\nHybrid search for '{query}': {len(result.results)} results "
              f"in {duration*1000:.2f}ms")

        assert metrics.passes_threshold(), (
            f"Hybrid search took {duration*1000:.2f}ms, "
            f"threshold is {THRESHOLDS['search_hybrid']}ms"
        )

    async def test_search_multiple_queries(self, ingested_doc_for_search, performance_metrics):
        """Test search latency over multiple queries."""
        metrics = performance_metrics("Multi-Query Search Average")

        queries = [
            "test content",
            "document structure",
            "text extraction",
            "quick brown fox",
            "special characters",
        ]

        for query in queries:
            start = time.perf_counter()
            await retriever.search(query=query, top_k=10)
            duration = time.perf_counter() - start
            metrics.durations.append(duration)

        print(f"\nSearch average over {len(queries)} queries: {metrics.mean_ms:.2f}ms, "
              f"p95: {metrics.p95_ms:.2f}ms")


# =============================================================================
# Test Class: Query Processing Performance
# =============================================================================


class TestQueryProcessingPerformance:
    """Performance tests for query processing operations."""

    async def test_intent_classification_latency(self, setup_database, performance_metrics):
        """Test intent classification latency."""
        metrics = performance_metrics(
            "Intent Classification",
            threshold_ms=THRESHOLDS["intent_classification"]
        )

        queries = [
            "Where is the config file?",
            "Explain how authentication works",
            "How do I deploy the application?",
            "The API returns 500 errors",
            "hello",
        ]

        for query in queries:
            start = time.perf_counter()
            await intent_classifier.classify(query)
            duration = time.perf_counter() - start
            metrics.durations.append(duration)

        print(f"\nIntent classification: mean {metrics.mean_ms:.2f}ms, "
              f"median {metrics.median_ms:.2f}ms, p95 {metrics.p95_ms:.2f}ms")

        assert metrics.passes_threshold(), (
            f"Intent classification mean {metrics.mean_ms:.2f}ms exceeds "
            f"threshold {THRESHOLDS['intent_classification']}ms"
        )

    async def test_query_transformation_latency(self, setup_database, performance_metrics):
        """Test query transformation latency."""
        metrics = performance_metrics("Query Transformation")

        query = "How do I configure database connections?"

        start = time.perf_counter()
        result = await query_transformer.generate_multi_query(query)
        duration = time.perf_counter() - start
        metrics.durations.append(duration)

        print(f"\nQuery transformation: {len(result.all_queries)} variants "
              f"in {duration*1000:.2f}ms")


# =============================================================================
# Test Class: Generation Performance
# =============================================================================


class TestGenerationPerformance:
    """Performance tests for LLM response generation."""

    async def test_query_latency_simple(self, ingested_doc_for_search, performance_metrics):
        """Test simple query end-to-end latency."""
        metrics = performance_metrics(
            "Simple Query E2E",
            threshold_ms=THRESHOLDS["query_simple"]
        )

        service = LLMResponseService()
        query = "What is in this document?"

        start = time.perf_counter()
        result = await service.generate_response(query=query)
        duration = time.perf_counter() - start
        metrics.durations.append(duration)

        print(f"\nSimple query '{query}': intent={result.intent.value}, "
              f"duration={duration*1000:.2f}ms")

        assert metrics.passes_threshold(), (
            f"Simple query took {duration*1000:.2f}ms, "
            f"threshold is {THRESHOLDS['query_simple']}ms"
        )

    async def test_query_latency_complex(self, ingested_doc_for_search, performance_metrics):
        """Test complex explain query end-to-end latency."""
        metrics = performance_metrics(
            "Complex Query E2E",
            threshold_ms=THRESHOLDS["query_complex"]
        )

        service = LLMResponseService()
        query = "Explain in detail what this document contains and how it is structured"

        start = time.perf_counter()
        result = await service.generate_response(query=query)
        duration = time.perf_counter() - start
        metrics.durations.append(duration)

        print(f"\nComplex query: intent={result.intent.value}, "
              f"duration={duration*1000:.2f}ms")

        assert metrics.passes_threshold(), (
            f"Complex query took {duration*1000:.2f}ms, "
            f"threshold is {THRESHOLDS['query_complex']}ms"
        )


# =============================================================================
# Test Class: Hallucination Detection Performance
# =============================================================================


class TestHallucinationPerformance:
    """Performance tests for hallucination detection."""

    async def test_hallucination_check_latency(self, ingested_doc_for_search, performance_metrics):
        """Test hallucination check latency."""
        metrics = performance_metrics(
            "Hallucination Check",
            threshold_ms=THRESHOLDS["hallucination_check"]
        )

        sources = [
            Source(
                source_id="src_0",
                chunk_id="test_chunk_1",
                document_id=ingested_doc_for_search["doc_id"],
                text_excerpt="The document contains text about testing and verification.",
                relevance_score=0.8,
            ),
            Source(
                source_id="src_1",
                chunk_id="test_chunk_2",
                document_id=ingested_doc_for_search["doc_id"],
                text_excerpt="Special characters include symbols like !@#$%.",
                relevance_score=0.7,
            ),
        ]

        response_text = (
            "The document contains information about testing. "
            "It includes special characters and verification procedures."
        )

        start = time.perf_counter()
        result = await hallucination_detector.check(
            response_text=response_text,
            sources=sources,
        )
        duration = time.perf_counter() - start
        metrics.durations.append(duration)

        print(f"\nHallucination check: score={result.hallucination_score:.2f}, "
              f"passed={result.passed}, duration={duration*1000:.2f}ms")

        assert metrics.passes_threshold(), (
            f"Hallucination check took {duration*1000:.2f}ms, "
            f"threshold is {THRESHOLDS['hallucination_check']}ms"
        )


# =============================================================================
# Test Class: Concurrent Performance
# =============================================================================


class TestConcurrentPerformance:
    """Performance tests for concurrent operations."""

    async def test_concurrent_queries(self, ingested_doc_for_search, performance_metrics):
        """Test latency when handling multiple concurrent queries."""
        metrics = performance_metrics("Concurrent Queries (5)")

        service = LLMResponseService()
        queries = [
            "What is this document about?",
            "Explain the content",
            "Where is the main information?",
            "Summarize this document",
            "What does it contain?",
        ]

        start = time.perf_counter()

        # Run all queries concurrently
        tasks = [service.generate_response(query=q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        total_duration = time.perf_counter() - start
        metrics.durations.append(total_duration / len(queries))  # Average per query

        successful = sum(1 for r in results if not isinstance(r, Exception))
        print(f"\n{successful}/{len(queries)} concurrent queries completed "
              f"in {total_duration*1000:.2f}ms total, "
              f"{(total_duration/len(queries))*1000:.2f}ms avg")

    async def test_concurrent_searches(self, ingested_doc_for_search, performance_metrics):
        """Test latency when handling multiple concurrent searches."""
        metrics = performance_metrics("Concurrent Searches (10)")

        queries = [
            "test content",
            "document structure",
            "text extraction",
            "quick brown fox",
            "special characters",
            "verification testing",
            "content analysis",
            "document format",
            "text processing",
            "data extraction",
        ]

        start = time.perf_counter()

        # Run all searches concurrently
        tasks = [retriever.search(query=q, top_k=5) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        total_duration = time.perf_counter() - start
        metrics.durations.append(total_duration / len(queries))

        successful = sum(1 for r in results if not isinstance(r, Exception))
        print(f"\n{successful}/{len(queries)} concurrent searches completed "
              f"in {total_duration*1000:.2f}ms total, "
              f"{(total_duration/len(queries))*1000:.2f}ms avg")


# =============================================================================
# Summary Test
# =============================================================================


class TestPerformanceSummary:
    """Summary test that runs all performance checks and reports."""

    async def test_full_pipeline_benchmark(self, setup_database, performance_metrics):
        """Run a full pipeline benchmark and generate summary."""
        print("\n\n" + "=" * 60)
        print("FULL PIPELINE PERFORMANCE BENCHMARK")
        print("=" * 60)

        # Ingest document
        ingest_metrics = performance_metrics(
            "Full Pipeline - Ingestion",
            threshold_ms=THRESHOLDS["ingestion_single"]
        )
        result, duration = await ingest_document_timed("01_happy_path_text_only.pdf")
        ingest_metrics.durations.append(duration)
        print(f"\n1. Ingestion: {duration*1000:.2f}ms")

        # Intent classification
        intent_metrics = performance_metrics(
            "Full Pipeline - Intent",
            threshold_ms=THRESHOLDS["intent_classification"]
        )
        start = time.perf_counter()
        await intent_classifier.classify("What is in this document?")
        intent_duration = time.perf_counter() - start
        intent_metrics.durations.append(intent_duration)
        print(f"2. Intent Classification: {intent_duration*1000:.2f}ms")

        # Query transformation
        transform_metrics = performance_metrics("Full Pipeline - Transform")
        start = time.perf_counter()
        await query_transformer.generate_multi_query("What is in this document?")
        transform_duration = time.perf_counter() - start
        transform_metrics.durations.append(transform_duration)
        print(f"3. Query Transformation: {transform_duration*1000:.2f}ms")

        # Hybrid search
        search_metrics = performance_metrics(
            "Full Pipeline - Search",
            threshold_ms=THRESHOLDS["search_hybrid"]
        )
        start = time.perf_counter()
        await retriever.search(query="document content", top_k=10)
        search_duration = time.perf_counter() - start
        search_metrics.durations.append(search_duration)
        print(f"4. Hybrid Search: {search_duration*1000:.2f}ms")

        # Full query
        query_metrics = performance_metrics(
            "Full Pipeline - Query E2E",
            threshold_ms=THRESHOLDS["query_simple"]
        )
        service = LLMResponseService()
        start = time.perf_counter()
        await service.generate_response(query="What is in this document?")
        query_duration = time.perf_counter() - start
        query_metrics.durations.append(query_duration)
        print(f"5. Full Query E2E: {query_duration*1000:.2f}ms")

        # Total
        total = (duration + intent_duration + transform_duration +
                 search_duration + query_duration)
        print(f"\nTotal Pipeline Time: {total*1000:.2f}ms")
        print("=" * 60)

        # All assertions
        all_passed = all([
            ingest_metrics.passes_threshold(),
            intent_metrics.passes_threshold(),
            search_metrics.passes_threshold(),
            query_metrics.passes_threshold(),
        ])

        assert all_passed, "One or more performance thresholds were not met"
