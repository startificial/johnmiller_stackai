"""
SQLAlchemy models for RAG pipeline persistence.

Document Storage Tables:
- Document: Master record for each processed document
- ExtractedPage: Page-level extracted text content
- DocumentChunk: Hierarchical chunks with parent-child relationships

BM25 Search Index Tables:
- BM25IndexedChunk: Links to DocumentChunk with BM25-specific data
- BM25Vocabulary: Maps terms to IDs and tracks document frequency
- BM25Posting: Inverted index mapping terms to chunks with term frequencies
- BM25CorpusStats: Singleton table for corpus-wide statistics
"""

from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    String,
    Text,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    UniqueConstraint,
    Enum,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from backend.app.db.base import Base


# =============================================================================
# Enums
# =============================================================================


class DocumentStatus(PyEnum):
    """Processing status for documents."""

    PENDING = "pending"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# Document Storage Models
# =============================================================================


class Document(Base):
    """
    Master record for each document processed through the RAG pipeline.

    Stores document metadata and processing status. Does not store the
    actual file content (PDF bytes).
    """

    __tablename__ = "documents"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    # File information
    filename = Column(String(512), nullable=False)
    file_hash = Column(String(64), nullable=True, index=True)  # SHA256 for dedup
    file_size = Column(BigInteger, nullable=True)  # bytes
    mime_type = Column(String(128), nullable=True)

    # Extraction results
    page_count = Column(Integer, nullable=True)
    total_char_count = Column(Integer, nullable=True)

    # Document metadata from extraction (title, author, subject, etc.)
    # Note: Using 'doc_metadata' to avoid conflict with SQLAlchemy's reserved 'metadata' attribute
    doc_metadata = Column(JSONB, default=dict)

    # Processing status
    status = Column(
        Enum(DocumentStatus),
        nullable=False,
        default=DocumentStatus.PENDING,
        index=True,
    )
    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relationships
    pages = relationship(
        "ExtractedPage",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="ExtractedPage.page_number",
    )
    chunks = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_documents_created_at", "created_at"),
        Index("ix_documents_status_created", "status", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Document(id='{self.id}', filename='{self.filename}', status='{self.status.value}')>"


class ExtractedPage(Base):
    """
    Stores extracted text content for each page of a document.

    Preserves page-level granularity for:
    - Page-specific retrieval
    - Debugging extraction issues
    - Re-chunking without re-extraction
    """

    __tablename__ = "extracted_pages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    page_number = Column(Integer, nullable=False)  # 1-indexed
    text = Column(Text, nullable=False)
    char_count = Column(Integer, nullable=False)

    # Relationship
    document = relationship("Document", back_populates="pages")

    __table_args__ = (
        UniqueConstraint("document_id", "page_number", name="uq_page_per_document"),
        Index("ix_extracted_pages_doc_page", "document_id", "page_number"),
    )

    def __repr__(self) -> str:
        return f"<ExtractedPage(document_id='{self.document_id}', page={self.page_number})>"


class DocumentChunk(Base):
    """
    Stores hierarchical chunks created from document text.

    Implements the "retrieve small, read big" pattern:
    - Level 0 (parent): Larger chunks for context during LLM generation
    - Level 1 (leaf): Smaller chunks for precise retrieval

    This is the source of truth for chunk content. BM25IndexedChunk
    references this table for search indexing.
    """

    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(String(255), unique=True, nullable=False, index=True)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Content
    text = Column(Text, nullable=False)
    char_count = Column(Integer, nullable=False)

    # Hierarchy
    level = Column(Integer, nullable=False, default=1)  # 0=parent, 1=leaf
    parent_id = Column(
        Integer,
        ForeignKey("document_chunks.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Position metadata
    page_number = Column(Integer, nullable=True)
    chunk_index = Column(Integer, nullable=False)  # Sequential index within document
    position = Column(Integer, nullable=True)  # Start character position in full text

    # Additional metadata (flexible)
    # Note: Using 'chunk_metadata' to avoid conflict with SQLAlchemy's reserved 'metadata' attribute
    chunk_metadata = Column(JSONB, default=dict)

    # Embedding for vector search (nullable until embedding service is implemented)
    # Using JSONB to store as array; can migrate to pgvector extension later
    embedding = Column(JSONB, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, server_default=func.now())

    # Relationships
    document = relationship("Document", back_populates="chunks")
    parent = relationship(
        "DocumentChunk",
        remote_side=[id],
        backref="children",
    )
    bm25_index = relationship(
        "BM25IndexedChunk",
        back_populates="document_chunk",
        uselist=False,
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_document_chunks_doc_level", "document_id", "level"),
        Index("ix_document_chunks_doc_page", "document_id", "page_number"),
    )

    @property
    def is_parent(self) -> bool:
        """Check if this chunk has children."""
        return self.level == 0

    @property
    def is_leaf(self) -> bool:
        """Check if this chunk is a leaf node."""
        return self.level == 1

    def __repr__(self) -> str:
        return f"<DocumentChunk(chunk_id='{self.chunk_id}', level={self.level})>"


# =============================================================================
# BM25 Search Index Models
# =============================================================================


class BM25IndexedChunk(Base):
    """
    BM25 search index entry for a document chunk.

    Links to DocumentChunk (required) and stores BM25-specific data.
    The actual text content is retrieved from the linked DocumentChunk.
    """

    __tablename__ = "bm25_indexed_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Required link to source chunk
    document_chunk_id = Column(
        Integer,
        ForeignKey("document_chunks.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # BM25-specific: token count after tokenization (may differ from char_count)
    token_count = Column(Integer, nullable=False)

    # Denormalized for query performance (avoids joins during search)
    chunk_id = Column(String(255), nullable=False, index=True)
    doc_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    # Relationships
    document_chunk = relationship("DocumentChunk", back_populates="bm25_index")
    postings = relationship(
        "BM25Posting",
        back_populates="chunk",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<BM25IndexedChunk(chunk_id='{self.chunk_id}', tokens={self.token_count})>"


class BM25Vocabulary(Base):
    """Vocabulary table mapping terms to IDs with document frequency."""

    __tablename__ = "bm25_vocabulary"

    id = Column(Integer, primary_key=True, autoincrement=True)
    term = Column(String(255), unique=True, nullable=False, index=True)
    document_frequency = Column(Integer, nullable=False, default=0)

    # Relationship to postings
    postings = relationship(
        "BM25Posting",
        back_populates="term_entry",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<BM25Vocabulary(term='{self.term}', df={self.document_frequency})>"


class BM25Posting(Base):
    """Inverted index: maps terms to chunks with term frequency."""

    __tablename__ = "bm25_postings"

    term_id = Column(
        Integer,
        ForeignKey("bm25_vocabulary.id", ondelete="CASCADE"),
        primary_key=True,
    )
    chunk_db_id = Column(
        Integer,
        ForeignKey("bm25_indexed_chunks.id", ondelete="CASCADE"),
        primary_key=True,
    )
    term_frequency = Column(Integer, nullable=False)

    # Relationships
    term_entry = relationship("BM25Vocabulary", back_populates="postings")
    chunk = relationship("BM25IndexedChunk", back_populates="postings")

    # Index for efficient lookups by term
    __table_args__ = (
        Index("ix_bm25_postings_term_id", "term_id"),
        Index("ix_bm25_postings_chunk_db_id", "chunk_db_id"),
    )

    def __repr__(self) -> str:
        return f"<BM25Posting(term_id={self.term_id}, chunk_db_id={self.chunk_db_id}, tf={self.term_frequency})>"


class BM25CorpusStats(Base):
    """Singleton table for corpus-wide BM25 statistics."""

    __tablename__ = "bm25_corpus_stats"

    id = Column(Integer, primary_key=True, default=1)
    total_documents = Column(Integer, nullable=False, default=0)
    avg_document_length = Column(Float, nullable=False, default=0.0)
    total_tokens = Column(Integer, nullable=False, default=0)

    # Ensure only one row exists
    __table_args__ = (
        UniqueConstraint("id", name="uq_bm25_corpus_stats_singleton"),
    )

    def __repr__(self) -> str:
        return f"<BM25CorpusStats(total_docs={self.total_documents}, avgdl={self.avg_document_length:.2f})>"
