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
    """Settings for Mistral AI embedding service."""

    # Mistral API key (required)
    api_key: str = field(
        default_factory=lambda: _get_env_str("MISTRAL_API_KEY", "")
    )

    # Model name for embeddings (mistral-embed produces 1024-dimensional vectors)
    model_name: str = field(
        default_factory=lambda: _get_env_str("EMBEDDING_MODEL", "mistral-embed")
    )

    # Embedding dimension (mistral-embed: 1024)
    dimension: int = field(
        default_factory=lambda: _get_env_int("EMBEDDING_DIMENSION", 1024)
    )

    # Batch size for embedding generation
    batch_size: int = field(
        default_factory=lambda: _get_env_int("EMBEDDING_BATCH_SIZE", 32)
    )

    # Maximum retries for API calls
    max_retries: int = field(
        default_factory=lambda: _get_env_int("EMBEDDING_MAX_RETRIES", 3)
    )

    # Timeout for API calls in seconds
    timeout: float = field(
        default_factory=lambda: _get_env_float("EMBEDDING_TIMEOUT", 60.0)
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
class QueryTransformSettings:
    """Settings for query transformation service."""

    # Mistral API key (shared with embedding service)
    api_key: str = field(
        default_factory=lambda: _get_env_str("MISTRAL_API_KEY", "")
    )

    # Model name for query generation (fast models for low latency)
    # Options: mistral-small-latest, open-mistral-7b, mistral-tiny
    model_name: str = field(
        default_factory=lambda: _get_env_str("QUERY_TRANSFORM_MODEL", "mistral-small-latest")
    )

    # Number of query variants to generate for multi-query (RAG-Fusion)
    num_queries: int = field(
        default_factory=lambda: _get_env_int("QUERY_TRANSFORM_NUM_QUERIES", 4)
    )

    # Temperature for generation (higher = more diverse queries)
    temperature: float = field(
        default_factory=lambda: _get_env_float("QUERY_TRANSFORM_TEMPERATURE", 0.7)
    )

    # Maximum tokens for generated queries
    max_tokens: int = field(
        default_factory=lambda: _get_env_int("QUERY_TRANSFORM_MAX_TOKENS", 256)
    )

    # Maximum retries for API calls
    max_retries: int = field(
        default_factory=lambda: _get_env_int("QUERY_TRANSFORM_MAX_RETRIES", 3)
    )

    # Timeout for API calls in seconds
    timeout: float = field(
        default_factory=lambda: _get_env_float("QUERY_TRANSFORM_TIMEOUT", 30.0)
    )


@dataclass
class RetrieverSettings:
    """Settings for hybrid retriever service."""

    # RRF (Reciprocal Rank Fusion) constant
    # Higher values give less weight to top results (typically 60)
    rrf_k: int = field(
        default_factory=lambda: _get_env_int("RETRIEVER_RRF_K", 60)
    )

    # Weight for semantic search results in fusion
    semantic_weight: float = field(
        default_factory=lambda: _get_env_float("RETRIEVER_SEMANTIC_WEIGHT", 1.0)
    )

    # Weight for keyword search results in fusion
    keyword_weight: float = field(
        default_factory=lambda: _get_env_float("RETRIEVER_KEYWORD_WEIGHT", 1.0)
    )

    # Default number of results to return
    default_top_k: int = field(
        default_factory=lambda: _get_env_int("RETRIEVER_DEFAULT_TOP_K", 10)
    )

    # Whether to expand results with parent chunk context by default
    expand_context: bool = field(
        default_factory=lambda: _get_env_bool("RETRIEVER_EXPAND_CONTEXT", True)
    )

    # Multiplier for internal search (fetch more candidates for fusion)
    search_multiplier: int = field(
        default_factory=lambda: _get_env_int("RETRIEVER_SEARCH_MULTIPLIER", 3)
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
    retriever: RetrieverSettings = field(default_factory=RetrieverSettings)
    query_transform: QueryTransformSettings = field(default_factory=QueryTransformSettings)

    # Application settings
    debug: bool = field(
        default_factory=lambda: _get_env_bool("DEBUG", False)
    )


# Singleton instance
settings = Settings()
