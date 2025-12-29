"""
Application Settings

Centralized configuration for all services in the RAG pipeline.
All configurable parameters should be defined here for easy management.
"""

import os
from dataclasses import dataclass, field
from typing import List


def _get_env_int(key: str, default: int) -> int:
    """Get integer from environment variable or return default."""
    value = os.getenv(key)
    if value is not None:
        try:
            return int(value)
        except ValueError:
            pass
    return default


def _get_env_str(key: str, default: str) -> str:
    """Get string from environment variable or return default."""
    return os.getenv(key, default)


def _get_env_bool(key: str, default: bool) -> bool:
    """Get boolean from environment variable or return default."""
    value = os.getenv(key)
    if value is not None:
        return value.lower() in ("true", "1", "yes")
    return default


def _get_env_float(key: str, default: float) -> float:
    """Get float from environment variable or return default."""
    value = os.getenv(key)
    if value is not None:
        try:
            return float(value)
        except ValueError:
            pass
    return default


@dataclass
class ChunkingSettings:
    """Settings for the structural chunker."""

    # Parent chunk settings (Level 0 - for context)
    parent_chunk_size: int = field(
        default_factory=lambda: _get_env_int("CHUNKING_PARENT_SIZE", 2000)
    )

    # Child chunk settings (Level 1 - for retrieval)
    child_chunk_size: int = field(
        default_factory=lambda: _get_env_int("CHUNKING_CHILD_SIZE", 500)
    )

    # Overlap between consecutive chunks
    chunk_overlap: int = field(
        default_factory=lambda: _get_env_int("CHUNKING_OVERLAP", 50)
    )

    # Minimum chunk size to create
    min_chunk_size: int = field(
        default_factory=lambda: _get_env_int("CHUNKING_MIN_SIZE", 100)
    )


@dataclass
class TextExtractionSettings:
    """Settings for text extraction service."""

    # Maximum file size in bytes (default: 50MB)
    max_file_size: int = field(
        default_factory=lambda: _get_env_int("EXTRACTION_MAX_FILE_SIZE", 50 * 1024 * 1024)
    )

    # Supported file extensions
    supported_extensions: List[str] = field(
        default_factory=lambda: [".pdf"]
    )

    # Whether to extract metadata from documents
    extract_metadata: bool = field(
        default_factory=lambda: _get_env_bool("EXTRACTION_METADATA", True)
    )


@dataclass
class EmbeddingSettings:
    """Settings for embedding service (future use)."""

    # Model name for embeddings
    model_name: str = field(
        default_factory=lambda: _get_env_str("EMBEDDING_MODEL", "text-embedding-3-small")
    )

    # Embedding dimension
    dimension: int = field(
        default_factory=lambda: _get_env_int("EMBEDDING_DIMENSION", 1536)
    )

    # Batch size for embedding generation
    batch_size: int = field(
        default_factory=lambda: _get_env_int("EMBEDDING_BATCH_SIZE", 100)
    )


@dataclass
class DatabaseSettings:
    """Settings for database connection."""

    # PostgreSQL connection URL (async)
    url: str = field(
        default_factory=lambda: _get_env_str(
            "DATABASE_URL",
            "postgresql+asyncpg://rag_user:rag_password@localhost:5432/rag_db"
        )
    )


@dataclass
class SearchSettings:
    """Settings for BM25 search indexer."""

    # BM25 k1 parameter: term frequency saturation (typically 1.2-2.0)
    bm25_k1: float = field(
        default_factory=lambda: _get_env_float("SEARCH_BM25_K1", 1.5)
    )

    # BM25 b parameter: document length normalization (0-1)
    bm25_b: float = field(
        default_factory=lambda: _get_env_float("SEARCH_BM25_B", 0.75)
    )

    # Tokenizer settings
    lowercase: bool = field(
        default_factory=lambda: _get_env_bool("SEARCH_LOWERCASE", True)
    )

    remove_stopwords: bool = field(
        default_factory=lambda: _get_env_bool("SEARCH_REMOVE_STOPWORDS", True)
    )

    min_token_length: int = field(
        default_factory=lambda: _get_env_int("SEARCH_MIN_TOKEN_LENGTH", 2)
    )

    # Search defaults
    default_top_k: int = field(
        default_factory=lambda: _get_env_int("SEARCH_DEFAULT_TOP_K", 10)
    )

    # Whether to index parent chunks in addition to leaf chunks
    index_parents: bool = field(
        default_factory=lambda: _get_env_bool("SEARCH_INDEX_PARENTS", False)
    )


@dataclass
class Settings:
    """Main application settings container."""

    # Service-specific settings
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    text_extraction: TextExtractionSettings = field(default_factory=TextExtractionSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    search: SearchSettings = field(default_factory=SearchSettings)

    # Application settings
    debug: bool = field(
        default_factory=lambda: _get_env_bool("DEBUG", False)
    )


# Singleton instance
settings = Settings()
