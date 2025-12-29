"""
Mistral AI Embedding Service

Generates embeddings for document chunks using Mistral's mistral-embed model.
Supports batch processing for efficiency and stores embeddings in PostgreSQL
using pgvector for similarity search.

The service implements:
- Batch embedding generation with configurable batch sizes
- Async database operations for storing embeddings
- Integration with ChunkedDocument from chunking service
- Error handling with retries for API resilience
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from mistralai import Mistral
from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.base import async_session_factory
from backend.app.db.models import DocumentChunk
from backend.app.services.chunking import Chunk, ChunkedDocument
from backend.app.settings import settings


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""

    pass


class EmbeddingConfigError(Exception):
    """Raised when embedding configuration is invalid."""

    pass


@dataclass
class EmbeddingResult:
    """Result of embedding a single chunk."""

    chunk_id: str
    embedding: np.ndarray
    token_count: int  # Approximate tokens used for this chunk

    def to_list(self) -> List[float]:
        """Convert embedding to Python list for serialization."""
        return self.embedding.tolist()


@dataclass
class EmbeddingBatchResult:
    """Result of embedding multiple chunks."""

    results: List[EmbeddingResult]
    total_tokens: int
    failed_chunks: List[str] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        """Number of successfully embedded chunks."""
        return len(self.results)

    @property
    def failure_count(self) -> int:
        """Number of failed chunks."""
        return len(self.failed_chunks)


class MistralEmbedder:
    """
    Mistral AI embedding generator.

    Uses the mistral-embed model to generate 1024-dimensional embeddings.
    Handles batching and API communication.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        """
        Initialize the Mistral embedder.

        Args:
            api_key: Mistral API key (defaults to settings)
            model: Model name (defaults to settings)
            batch_size: Batch size for embedding (defaults to settings)
            max_retries: Max API retries (defaults to settings)
            timeout: API timeout in seconds (defaults to settings)
        """
        embedding_settings = settings.embedding

        self.api_key = api_key or embedding_settings.api_key
        self.model = model or embedding_settings.model_name
        self.batch_size = batch_size or embedding_settings.batch_size
        self.max_retries = max_retries or embedding_settings.max_retries
        self.timeout = timeout or embedding_settings.timeout
        self.dimension = embedding_settings.dimension

        if not self.api_key:
            raise EmbeddingConfigError(
                "MISTRAL_API_KEY environment variable is required"
            )

        # Initialize Mistral client
        self._client = Mistral(api_key=self.api_key)

    async def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of numpy arrays containing embeddings

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not texts:
            return []

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = await self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def _embed_batch(
        self, texts: List[str], retry_count: int = 0
    ) -> List[np.ndarray]:
        """
        Embed a single batch of texts with retry logic.

        Args:
            texts: Batch of texts to embed
            retry_count: Current retry attempt

        Returns:
            List of embeddings for the batch
        """
        try:
            # Mistral SDK is synchronous, run in thread pool
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.embeddings.create(
                    model=self.model,
                    inputs=texts,
                ),
            )

            # Extract embeddings from response
            embeddings = [
                np.array(item.embedding, dtype=np.float32) for item in response.data
            ]

            return embeddings

        except Exception as e:
            if retry_count < self.max_retries:
                # Exponential backoff
                wait_time = 2**retry_count
                await asyncio.sleep(wait_time)
                return await self._embed_batch(texts, retry_count + 1)
            else:
                raise EmbeddingError(
                    f"Failed to generate embeddings after {self.max_retries} retries: {e}"
                ) from e

    async def embed_chunk(self, chunk: Chunk) -> EmbeddingResult:
        """
        Generate embedding for a single chunk.

        Args:
            chunk: Chunk to embed

        Returns:
            EmbeddingResult with embedding and metadata
        """
        embeddings = await self.embed_texts([chunk.text])

        return EmbeddingResult(
            chunk_id=chunk.id,
            embedding=embeddings[0],
            token_count=len(chunk.text.split()),  # Approximate token count
        )

    async def embed_chunks(self, chunks: List[Chunk]) -> EmbeddingBatchResult:
        """
        Generate embeddings for multiple chunks.

        Args:
            chunks: List of chunks to embed

        Returns:
            EmbeddingBatchResult with all results
        """
        if not chunks:
            return EmbeddingBatchResult(results=[], total_tokens=0)

        texts = [chunk.text for chunk in chunks]

        try:
            embeddings = await self.embed_texts(texts)

            results = [
                EmbeddingResult(
                    chunk_id=chunk.id,
                    embedding=embedding,
                    token_count=len(chunk.text.split()),
                )
                for chunk, embedding in zip(chunks, embeddings)
            ]

            total_tokens = sum(r.token_count for r in results)

            return EmbeddingBatchResult(
                results=results,
                total_tokens=total_tokens,
            )

        except EmbeddingError:
            # If batch fails, mark all chunks as failed
            return EmbeddingBatchResult(
                results=[],
                total_tokens=0,
                failed_chunks=[chunk.id for chunk in chunks],
            )


class EmbeddingService:
    """
    High-level embedding service for RAG pipeline.

    Orchestrates embedding generation and database storage.
    Integrates with ChunkedDocument from chunking service.
    """

    def __init__(self, embedder: Optional[MistralEmbedder] = None):
        """
        Initialize the embedding service.

        Args:
            embedder: Optional MistralEmbedder instance. If not provided,
                     one will be created when first needed (lazy initialization).
        """
        self._embedder = embedder

    @property
    def embedder(self) -> MistralEmbedder:
        """Get or create the embedder instance (lazy initialization)."""
        if self._embedder is None:
            self._embedder = MistralEmbedder()
        return self._embedder

    async def embed_document(
        self,
        chunked_doc: ChunkedDocument,
        embed_parents: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate embeddings for all chunks in a document and store in database.

        Args:
            chunked_doc: ChunkedDocument from chunking service
            embed_parents: Whether to embed parent chunks (default: only leaf chunks)

        Returns:
            Dictionary with embedding statistics
        """
        # Determine which chunks to embed
        chunks_to_embed = chunked_doc.leaf_chunks
        if embed_parents:
            chunks_to_embed = chunked_doc.chunks

        if not chunks_to_embed:
            return {
                "document_id": chunked_doc.doc_id,
                "chunks_embedded": 0,
                "total_tokens": 0,
                "failed_chunks": [],
            }

        # Generate embeddings
        batch_result = await self.embedder.embed_chunks(chunks_to_embed)

        # Store embeddings in database
        stored_count = await self._store_embeddings(chunked_doc.doc_id, batch_result.results)

        return {
            "document_id": chunked_doc.doc_id,
            "chunks_embedded": stored_count,
            "total_tokens": batch_result.total_tokens,
            "failed_chunks": batch_result.failed_chunks,
        }

    async def _store_embeddings(
        self,
        doc_id: str,
        results: List[EmbeddingResult],
    ) -> int:
        """
        Store embeddings in the database.

        Args:
            doc_id: Document identifier
            results: List of EmbeddingResult objects

        Returns:
            Number of embeddings stored
        """
        if not results:
            return 0

        async with async_session_factory() as session:
            stored_count = 0

            for result in results:
                # Update the DocumentChunk with the embedding
                stmt = (
                    update(DocumentChunk)
                    .where(DocumentChunk.chunk_id == result.chunk_id)
                    .values(embedding=result.embedding.tolist())
                )

                update_result = await session.execute(stmt)
                if update_result.rowcount > 0:
                    stored_count += 1

            await session.commit()
            return stored_count

    async def embed_chunks_only(
        self,
        chunks: List[Chunk],
    ) -> EmbeddingBatchResult:
        """
        Generate embeddings for chunks without storing.

        Useful for query embedding or when storage is handled separately.

        Args:
            chunks: List of chunks to embed

        Returns:
            EmbeddingBatchResult with embeddings
        """
        return await self.embedder.embed_chunks(chunks)

    async def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.

        Args:
            query: Search query text

        Returns:
            Numpy array with query embedding
        """
        embeddings = await self.embedder.embed_texts([query])
        return embeddings[0]

    async def get_chunk_embeddings(
        self,
        chunk_ids: List[str],
    ) -> Dict[str, np.ndarray]:
        """
        Retrieve embeddings for chunks from database.

        Args:
            chunk_ids: List of chunk IDs to retrieve

        Returns:
            Dictionary mapping chunk_id to embedding
        """
        async with async_session_factory() as session:
            result = await session.execute(
                select(DocumentChunk.chunk_id, DocumentChunk.embedding)
                .where(DocumentChunk.chunk_id.in_(chunk_ids))
                .where(DocumentChunk.embedding.isnot(None))
            )

            return {
                row.chunk_id: np.array(row.embedding, dtype=np.float32) for row in result
            }

    async def similarity_search(
        self,
        query: str,
        top_k: int = 10,
        filter_doc_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.

        Uses cosine distance for similarity ranking.

        Args:
            query: Search query text
            top_k: Number of results to return
            filter_doc_ids: Optional list of doc_ids to restrict search

        Returns:
            List of search results with chunk data and scores
        """
        # Generate query embedding
        query_embedding = await self.embed_query(query)

        async with async_session_factory() as session:
            # Convert embedding to PostgreSQL array format
            embedding_str = "[" + ",".join(map(str, query_embedding.tolist())) + "]"

            # Build the SQL query using pgvector's <=> operator for cosine distance
            sql = f"""
                SELECT
                    chunk_id,
                    document_id,
                    text,
                    level,
                    page_number,
                    chunk_index,
                    chunk_metadata,
                    embedding <=> '{embedding_str}'::vector AS distance
                FROM document_chunks
                WHERE embedding IS NOT NULL
            """

            if filter_doc_ids:
                uuid_list = ", ".join(f"'{d}'" for d in filter_doc_ids)
                sql += f" AND document_id IN ({uuid_list})"

            sql += f" ORDER BY distance LIMIT {top_k}"

            result = await session.execute(text(sql))

            return [
                {
                    "chunk_id": row.chunk_id,
                    "document_id": str(row.document_id),
                    "text": row.text,
                    "level": row.level,
                    "page_number": row.page_number,
                    "chunk_index": row.chunk_index,
                    "metadata": row.chunk_metadata or {},
                    "distance": float(row.distance),
                    "similarity": 1 - float(row.distance),  # Convert distance to similarity
                }
                for row in result
            ]


# Lazy-initialized singleton instance
# The embedder is created on first use to avoid requiring API key at import time
embedding_service = EmbeddingService()
