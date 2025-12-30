"""
Pytest configuration and fixtures for integration tests.

Provides:
- Database initialization and cleanup
- Async session management
- Document ingestion helpers
- Test document paths
"""

import asyncio
import os
from pathlib import Path
from typing import AsyncGenerator, Generator

# Load environment variables from .env file BEFORE importing app modules
from dotenv import load_dotenv

# Load from project root .env
_project_root = Path(__file__).parent.parent.parent
_env_path = _project_root / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from backend.app.db.base import Base, engine, async_session_factory, init_db, drop_db
from backend.app.db.models import Document, DocumentChunk  # noqa: F401 - needed for metadata


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
# Event Loop Configuration
# =============================================================================


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Database Fixtures
# =============================================================================


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Provide a clean database session for each test.

    Creates all tables before the test and drops them after.
    This ensures each test runs with a clean database state.
    """
    # Initialize database (creates tables)
    await init_db()

    async with async_session_factory() as session:
        yield session

    # Clean up after test
    await drop_db()


@pytest_asyncio.fixture(scope="module")
async def db_session_module() -> AsyncGenerator[AsyncSession, None]:
    """
    Provide a database session shared across a test module.

    Use this when tests in a module need to share state
    (e.g., one test ingests, another queries).
    """
    # Initialize database (creates tables)
    await init_db()

    async with async_session_factory() as session:
        yield session

    # Clean up after all tests in module
    await drop_db()


# =============================================================================
# Document Ingestion Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def ingest_test_document(db_session: AsyncSession):
    """
    Factory fixture to ingest a test document.

    Usage:
        async def test_something(ingest_test_document):
            doc_id = await ingest_test_document("01_happy_path_text_only.pdf")
    """
    from backend.app.services.text_extraction import text_extraction_service
    from backend.app.services.chunking import chunking_service
    from backend.app.services.embedding import embedding_service
    from backend.app.services.search_indexer import search_indexer

    async def _ingest(filename: str) -> str:
        """Ingest a document and return its ID."""
        file_path = get_test_document_path(filename)

        # Step 1: Extract text
        extracted_doc = await text_extraction_service.extract_and_persist(
            file_path=file_path,
        )

        # Step 2: Chunk document
        chunked_doc = await chunking_service.chunk_and_persist(
            document=extracted_doc,
            doc_id=extracted_doc.doc_id,
        )

        # Step 3: Generate embeddings (for leaf chunks)
        await embedding_service.embed_document(
            chunked_doc=chunked_doc,
            embed_parents=False,
        )

        # Step 4: Index for BM25 search
        await search_indexer.index_chunked_document(chunked_doc)

        return str(extracted_doc.doc_id)

    return _ingest


# =============================================================================
# LLM Service Fixtures
# =============================================================================


@pytest.fixture
def llm_response_service():
    """Provide the LLM response service singleton."""
    from backend.app.services.llm_response.service import llm_response_service
    return llm_response_service


# =============================================================================
# Performance Testing Utilities
# =============================================================================

import time
from dataclasses import dataclass, field
from typing import List, Optional
from contextlib import contextmanager
import statistics


@dataclass
class PerformanceMetrics:
    """Container for performance test metrics."""

    operation: str
    durations: List[float] = field(default_factory=list)
    threshold_ms: Optional[float] = None

    @property
    def count(self) -> int:
        return len(self.durations)

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.durations) * 1000 if self.durations else 0

    @property
    def median_ms(self) -> float:
        return statistics.median(self.durations) * 1000 if self.durations else 0

    @property
    def p95_ms(self) -> float:
        if len(self.durations) < 2:
            return self.mean_ms
        sorted_durations = sorted(self.durations)
        idx = int(len(sorted_durations) * 0.95)
        return sorted_durations[idx] * 1000

    @property
    def p99_ms(self) -> float:
        if len(self.durations) < 2:
            return self.mean_ms
        sorted_durations = sorted(self.durations)
        idx = int(len(sorted_durations) * 0.99)
        return sorted_durations[idx] * 1000

    @property
    def min_ms(self) -> float:
        return min(self.durations) * 1000 if self.durations else 0

    @property
    def max_ms(self) -> float:
        return max(self.durations) * 1000 if self.durations else 0

    def passes_threshold(self) -> bool:
        """Check if mean duration is under threshold."""
        if self.threshold_ms is None:
            return True
        return self.mean_ms < self.threshold_ms

    def to_dict(self) -> dict:
        return {
            "operation": self.operation,
            "count": self.count,
            "mean_ms": round(self.mean_ms, 2),
            "median_ms": round(self.median_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "threshold_ms": self.threshold_ms,
            "passes": self.passes_threshold(),
        }

    def __str__(self) -> str:
        status = "PASS" if self.passes_threshold() else "FAIL"
        threshold_str = f" (threshold: {self.threshold_ms}ms)" if self.threshold_ms else ""
        return (
            f"{self.operation}: {self.mean_ms:.2f}ms mean, "
            f"{self.median_ms:.2f}ms median, {self.p95_ms:.2f}ms p95{threshold_str} [{status}]"
        )


@contextmanager
def measure_time():
    """Context manager to measure execution time."""
    start = time.perf_counter()
    result = {"duration": 0.0}
    try:
        yield result
    finally:
        result["duration"] = time.perf_counter() - start


@pytest.fixture
def performance_metrics():
    """Factory fixture to create performance metrics."""
    metrics_list = []

    def _create(operation: str, threshold_ms: Optional[float] = None) -> PerformanceMetrics:
        metrics = PerformanceMetrics(operation=operation, threshold_ms=threshold_ms)
        metrics_list.append(metrics)
        return metrics

    yield _create

    # Print summary at end of test
    print("\n\n=== Performance Summary ===")
    for m in metrics_list:
        print(m)


# =============================================================================
# Marker Registration
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires DB + API keys)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance benchmark"
    )
