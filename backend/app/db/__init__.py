"""
Database module for RAG pipeline persistence.

Provides SQLAlchemy async setup and models for:
- Document storage (documents, pages, chunks)
- BM25 search index (vocabulary, postings, stats)
"""

from backend.app.db.base import Base, engine, async_session_factory, get_session, init_db
from backend.app.db.models import (
    # Enums
    DocumentStatus,
    # Document storage models
    Document,
    ExtractedPage,
    DocumentChunk,
    # BM25 search index models
    BM25IndexedChunk,
    BM25Vocabulary,
    BM25Posting,
    BM25CorpusStats,
)

__all__ = [
    # Base and session
    "Base",
    "engine",
    "async_session_factory",
    "get_session",
    "init_db",
    # Enums
    "DocumentStatus",
    # Document storage models
    "Document",
    "ExtractedPage",
    "DocumentChunk",
    # BM25 search index models
    "BM25IndexedChunk",
    "BM25Vocabulary",
    "BM25Posting",
    "BM25CorpusStats",
]
