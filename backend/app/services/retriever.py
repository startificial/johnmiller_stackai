"""
Hybrid Retriever Service

Implements hybrid search combining semantic (vector) and keyword (BM25) search
using Reciprocal Rank Fusion (RRF) for result merging. Supports parent chunk
context expansion for the "retrieve small, read big" pattern.

Search Flow:
1. Semantic search: Embed query, find nearest vectors in pgvector (leaf chunks)
2. Keyword search: BM25 ranking from search_indexer
3. Merge results using Reciprocal Rank Fusion (RRF)
4. Fetch parent chunks for context expansion
5. Return top-k results
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.base import async_session_factory
from backend.app.db.models import DocumentChunk
from backend.app.services.embedding import EmbeddingService, embedding_service
from backend.app.services.search_indexer import SearchIndexer, search_indexer
from backend.app.settings import settings


class RetrieverError(Exception):
    """Base exception for retriever errors."""

    pass


class NoEmbeddingsError(RetrieverError):
    """Raised when no embeddings exist in the database."""

    pass


class NoIndexError(RetrieverError):
    """Raised when no BM25 index exists."""

    pass


@dataclass
class RetrievalResult:
    """Single retrieval result with scores and optional parent context."""

    chunk_id: str
    document_id: str
    text: str
    level: int
    page_number: Optional[int]
    chunk_index: int

    # Scores from individual search methods
    semantic_score: Optional[float] = None  # Cosine similarity (0-1)
    keyword_score: Optional[float] = None  # Normalized BM25 (0-1)

    # Combined RRF score
    rrf_score: float = 0.0

    # Rank positions (1-indexed, None if not in that result set)
    semantic_rank: Optional[int] = None
    keyword_rank: Optional[int] = None

    # Parent context expansion
    parent_text: Optional[str] = None
    parent_chunk_id: Optional[str] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "level": self.level,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "semantic_score": self.semantic_score,
            "keyword_score": self.keyword_score,
            "rrf_score": self.rrf_score,
            "semantic_rank": self.semantic_rank,
            "keyword_rank": self.keyword_rank,
            "parent_text": self.parent_text,
            "parent_chunk_id": self.parent_chunk_id,
            "metadata": self.metadata,
        }


@dataclass
class RetrievalResponse:
    """Full retrieval response with results and metadata."""

    results: List[RetrievalResult]
    query: str
    total_semantic_results: int
    total_keyword_results: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "results": [r.to_dict() for r in self.results],
            "query": self.query,
            "total_semantic_results": self.total_semantic_results,
            "total_keyword_results": self.total_keyword_results,
            "result_count": len(self.results),
        }


@dataclass
class _ScoredChunk:
    """Internal representation of a scored chunk from a single search method."""

    chunk_id: str
    document_id: str
    text: str
    level: int
    page_number: Optional[int]
    chunk_index: int
    score: float  # Raw score from search method
    normalized_score: float  # Score normalized to 0-1
    metadata: Dict[str, Any]


class HybridRetriever:
    """
    Hybrid retrieval combining semantic and keyword search with RRF fusion.

    Uses EmbeddingService for semantic search and SearchIndexer for keyword
    search. Results are merged using Reciprocal Rank Fusion (RRF) algorithm.
    """

    def __init__(
        self,
        embedding_svc: Optional[EmbeddingService] = None,
        search_idx: Optional[SearchIndexer] = None,
    ):
        """
        Initialize the hybrid retriever.

        Args:
            embedding_svc: EmbeddingService instance (defaults to singleton)
            search_idx: SearchIndexer instance (defaults to singleton)
        """
        self._embedding_service = embedding_svc or embedding_service
        self._search_indexer = search_idx or search_indexer
        self._settings = settings.retriever

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_doc_ids: Optional[List[str]] = None,
        expand_context: Optional[bool] = None,
    ) -> RetrievalResponse:
        """
        Perform hybrid search combining semantic and keyword search.

        Args:
            query: Search query text
            top_k: Number of results to return (default from settings)
            filter_doc_ids: Optional list of document IDs to restrict search
            expand_context: Whether to fetch parent chunks (default from settings)

        Returns:
            RetrievalResponse with ranked results

        Raises:
            NoEmbeddingsError: If no embeddings exist in the database
            NoIndexError: If no BM25 index exists
            RetrieverError: For other retrieval errors
        """
        if top_k is None:
            top_k = self._settings.default_top_k
        if expand_context is None:
            expand_context = self._settings.expand_context

        # Fetch more candidates for better fusion results
        internal_top_k = top_k * self._settings.search_multiplier

        # Run both searches in parallel
        semantic_results, keyword_results = await asyncio.gather(
            self._semantic_search(query, internal_top_k, filter_doc_ids),
            self._keyword_search(query, internal_top_k, filter_doc_ids),
        )

        # Validate we have embeddings
        if not semantic_results:
            # Check if there are any embeddings at all
            has_embeddings = await self._check_embeddings_exist(filter_doc_ids)
            if not has_embeddings:
                raise NoEmbeddingsError(
                    "No embeddings found in database. Ensure documents have been "
                    "embedded using the EmbeddingService before searching."
                )

        # Validate we have BM25 index
        if not keyword_results:
            has_index = await self._check_index_exists(filter_doc_ids)
            if not has_index:
                raise NoIndexError(
                    "No BM25 index found. Ensure documents have been indexed "
                    "using the SearchIndexer before searching."
                )

        # Merge results using RRF
        fused_results = self._rrf_fusion(semantic_results, keyword_results)

        # Take top_k results
        fused_results = fused_results[:top_k]

        # Expand with parent context if requested
        if expand_context and fused_results:
            fused_results = await self._expand_parent_context(fused_results)

        return RetrievalResponse(
            results=fused_results,
            query=query,
            total_semantic_results=len(semantic_results),
            total_keyword_results=len(keyword_results),
        )

    async def _semantic_search(
        self,
        query: str,
        top_k: int,
        filter_doc_ids: Optional[List[str]] = None,
    ) -> List[_ScoredChunk]:
        """
        Perform semantic search using vector similarity.

        Args:
            query: Search query text
            top_k: Number of results to return
            filter_doc_ids: Optional document ID filter

        Returns:
            List of scored chunks from semantic search
        """
        try:
            # Use the embedding service's similarity search
            results = await self._embedding_service.similarity_search(
                query=query,
                top_k=top_k,
                filter_doc_ids=filter_doc_ids,
            )

            # Convert to internal format
            scored_chunks = []
            for result in results:
                # Similarity is already 0-1 (1 - cosine_distance)
                similarity = result.get("similarity", 0.0)

                scored_chunks.append(
                    _ScoredChunk(
                        chunk_id=result["chunk_id"],
                        document_id=result["document_id"],
                        text=result["text"],
                        level=result.get("level", 1),
                        page_number=result.get("page_number"),
                        chunk_index=result.get("chunk_index", 0),
                        score=similarity,
                        normalized_score=similarity,  # Already 0-1
                        metadata=result.get("metadata", {}),
                    )
                )

            return scored_chunks

        except Exception as e:
            # Log but don't fail - return empty results
            if settings.debug:
                print(f"Semantic search error: {e}")
            return []

    async def _keyword_search(
        self,
        query: str,
        top_k: int,
        filter_doc_ids: Optional[List[str]] = None,
    ) -> List[_ScoredChunk]:
        """
        Perform keyword search using BM25.

        Args:
            query: Search query text
            top_k: Number of results to return
            filter_doc_ids: Optional document ID filter

        Returns:
            List of scored chunks from keyword search
        """
        try:
            # Use the search indexer
            results = await self._search_indexer.search(
                query=query,
                top_k=top_k,
                filter_doc_ids=filter_doc_ids,
            )

            # Convert to internal format
            scored_chunks = []
            for result in results:
                scored_chunks.append(
                    _ScoredChunk(
                        chunk_id=result.chunk_id,
                        document_id=result.doc_id,
                        text=result.text,
                        level=result.metadata.get("level", 1),
                        page_number=result.metadata.get("page_number"),
                        chunk_index=result.metadata.get("chunk_index", 0),
                        score=result.score,
                        normalized_score=result.bm25_score,  # Already normalized 0-1
                        metadata=result.metadata,
                    )
                )

            return scored_chunks

        except Exception as e:
            if settings.debug:
                print(f"Keyword search error: {e}")
            return []

    def _rrf_fusion(
        self,
        semantic_results: List[_ScoredChunk],
        keyword_results: List[_ScoredChunk],
    ) -> List[RetrievalResult]:
        """
        Merge results using Reciprocal Rank Fusion (RRF).

        RRF formula: score(d) = Σ (weight / (k + rank(d)))

        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search

        Returns:
            Merged and ranked RetrievalResult list
        """
        k = self._settings.rrf_k
        semantic_weight = self._settings.semantic_weight
        keyword_weight = self._settings.keyword_weight

        # Build lookup maps by chunk_id
        # Maps: chunk_id -> (rank, scored_chunk)
        semantic_map: Dict[str, tuple[int, _ScoredChunk]] = {}
        for rank, chunk in enumerate(semantic_results, start=1):
            semantic_map[chunk.chunk_id] = (rank, chunk)

        keyword_map: Dict[str, tuple[int, _ScoredChunk]] = {}
        for rank, chunk in enumerate(keyword_results, start=1):
            keyword_map[chunk.chunk_id] = (rank, chunk)

        # Get all unique chunk IDs
        all_chunk_ids = set(semantic_map.keys()) | set(keyword_map.keys())

        # Calculate RRF scores
        fused_results: List[RetrievalResult] = []

        for chunk_id in all_chunk_ids:
            rrf_score = 0.0
            semantic_rank = None
            keyword_rank = None
            semantic_score = None
            keyword_score = None

            # Get the chunk data from whichever result set has it
            chunk_data = None

            if chunk_id in semantic_map:
                rank, chunk = semantic_map[chunk_id]
                semantic_rank = rank
                semantic_score = chunk.normalized_score
                rrf_score += semantic_weight / (k + rank)
                chunk_data = chunk

            if chunk_id in keyword_map:
                rank, chunk = keyword_map[chunk_id]
                keyword_rank = rank
                keyword_score = chunk.normalized_score
                rrf_score += keyword_weight / (k + rank)
                if chunk_data is None:
                    chunk_data = chunk

            # Create result
            result = RetrievalResult(
                chunk_id=chunk_data.chunk_id,
                document_id=chunk_data.document_id,
                text=chunk_data.text,
                level=chunk_data.level,
                page_number=chunk_data.page_number,
                chunk_index=chunk_data.chunk_index,
                semantic_score=semantic_score,
                keyword_score=keyword_score,
                rrf_score=rrf_score,
                semantic_rank=semantic_rank,
                keyword_rank=keyword_rank,
                metadata=chunk_data.metadata,
            )
            fused_results.append(result)

        # Sort by RRF score descending
        fused_results.sort(key=lambda x: x.rrf_score, reverse=True)

        return fused_results

    async def _expand_parent_context(
        self,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """
        Expand results with parent chunk context.

        For each leaf chunk (level=1), fetches its parent chunk (level=0)
        to provide additional context for the LLM.

        Args:
            results: List of retrieval results

        Returns:
            Results with parent_text and parent_chunk_id populated
        """
        # Collect chunk IDs that need parent lookup
        chunk_ids = [r.chunk_id for r in results if r.level == 1]

        if not chunk_ids:
            return results

        async with async_session_factory() as session:
            # Query for chunks and their parents
            result = await session.execute(
                select(
                    DocumentChunk.chunk_id,
                    DocumentChunk.parent_id,
                )
                .where(DocumentChunk.chunk_id.in_(chunk_ids))
                .where(DocumentChunk.parent_id.isnot(None))
            )

            # Map chunk_id -> parent database id
            chunk_to_parent_id: Dict[str, int] = {}
            parent_db_ids = set()

            for row in result:
                chunk_to_parent_id[row.chunk_id] = row.parent_id
                parent_db_ids.add(row.parent_id)

            if not parent_db_ids:
                return results

            # Fetch parent chunks
            parent_result = await session.execute(
                select(
                    DocumentChunk.id,
                    DocumentChunk.chunk_id,
                    DocumentChunk.text,
                )
                .where(DocumentChunk.id.in_(parent_db_ids))
            )

            # Map parent db id -> (chunk_id, text)
            parent_data: Dict[int, tuple[str, str]] = {}
            for row in parent_result:
                parent_data[row.id] = (row.chunk_id, row.text)

            # Update results with parent context
            for r in results:
                if r.chunk_id in chunk_to_parent_id:
                    parent_db_id = chunk_to_parent_id[r.chunk_id]
                    if parent_db_id in parent_data:
                        parent_chunk_id, parent_text = parent_data[parent_db_id]
                        r.parent_chunk_id = parent_chunk_id
                        r.parent_text = parent_text

        return results

    async def _check_embeddings_exist(
        self,
        filter_doc_ids: Optional[List[str]] = None,
    ) -> bool:
        """Check if any embeddings exist in the database."""
        async with async_session_factory() as session:
            sql = """
                SELECT EXISTS(
                    SELECT 1 FROM document_chunks
                    WHERE embedding IS NOT NULL
                    AND level = 1
            """
            if filter_doc_ids:
                uuid_list = ", ".join(f"'{d}'" for d in filter_doc_ids)
                sql += f" AND document_id IN ({uuid_list})"
            sql += ")"

            result = await session.execute(text(sql))
            return result.scalar()

    async def _check_index_exists(
        self,
        filter_doc_ids: Optional[List[str]] = None,
    ) -> bool:
        """Check if any BM25 index entries exist."""
        async with async_session_factory() as session:
            sql = "SELECT EXISTS(SELECT 1 FROM bm25_indexed_chunks"
            if filter_doc_ids:
                uuid_list = ", ".join(f"'{d}'" for d in filter_doc_ids)
                sql += f" WHERE doc_id IN ({uuid_list})"
            sql += ")"

            result = await session.execute(text(sql))
            return result.scalar()


# Singleton instance for easy imports
retriever = HybridRetriever()
