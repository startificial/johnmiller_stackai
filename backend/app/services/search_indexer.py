"""
BM25 Search Indexer Service

Implements BM25 (Best Matching 25) ranking algorithm from scratch for keyword-based
retrieval in a RAG pipeline. Integrates with PostgreSQL for persistence.

BM25 is a probabilistic ranking function that scores documents based on:
- Term frequency (TF) in the document
- Inverse document frequency (IDF) across the corpus
- Document length normalization

Parameters:
- k1: Term frequency saturation parameter (typically 1.2-2.0)
- b: Document length normalization (0 = no normalization, 1 = full normalization)
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any

from sqlalchemy import select, delete, func, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert

from uuid import UUID

from backend.app.db.base import async_session_factory
from backend.app.db.models import (
    DocumentChunk,
    BM25IndexedChunk,
    BM25Vocabulary,
    BM25Posting,
    BM25CorpusStats,
)
from backend.app.services.chunking import Chunk, ChunkedDocument
from backend.app.settings import settings


# Default English stopwords
DEFAULT_STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "this", "but", "they", "have",
    "had", "what", "when", "where", "who", "which", "why", "how", "all",
    "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "can", "just", "should", "now", "or", "if", "then",
    "also", "been", "could", "would", "there", "their", "them", "these",
    "those", "your", "our", "his", "her", "my", "i", "you", "we", "do",
    "does", "did", "doing", "done", "am", "being", "about", "into",
    "through", "during", "before", "after", "above", "below", "up",
    "down", "out", "off", "over", "under", "again", "further", "once",
    "here", "any", "while", "because", "until", "between", "s", "t",
    "d", "ll", "ve", "re", "m", "don", "didn", "doesn", "won", "wasn",
    "weren", "hasn", "haven", "hadn", "isn", "aren", "couldn", "wouldn",
    "shouldn", "ain", "let", "us", "me", "him", "she", "itself", "myself",
    "yourself", "himself", "herself", "ourselves", "themselves",
}


@dataclass
class SearchResult:
    """Represents a search result with BM25 score."""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    bm25_score: float  # Normalized 0-1 for hybrid fusion
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "score": self.score,
            "bm25_score": self.bm25_score,
            "metadata": self.metadata,
        }


class Tokenizer:
    """
    Configurable text tokenizer for BM25 indexing and search.

    Performs:
    - Lowercasing (configurable)
    - Punctuation removal
    - Whitespace splitting
    - Stopword filtering (configurable)
    - Minimum token length filtering
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_stopwords: bool = True,
        stopwords: Optional[Set[str]] = None,
        min_token_length: int = 2,
    ):
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS
        self.min_token_length = min_token_length

        # Regex pattern for tokenization: split on non-alphanumeric
        self._token_pattern = re.compile(r"[a-zA-Z0-9]+")

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into terms.

        Args:
            text: Input text to tokenize

        Returns:
            List of tokens after processing
        """
        if not text:
            return []

        # Lowercase if configured
        if self.lowercase:
            text = text.lower()

        # Extract tokens using regex
        tokens = self._token_pattern.findall(text)

        # Filter by minimum length
        tokens = [t for t in tokens if len(t) >= self.min_token_length]

        # Remove stopwords if configured
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]

        return tokens

    def get_term_frequencies(self, text: str) -> Dict[str, int]:
        """
        Get term frequency counts for text.

        Args:
            text: Input text

        Returns:
            Dictionary mapping terms to their frequencies
        """
        tokens = self.tokenize(text)
        return dict(Counter(tokens))


class BM25Index:
    """
    BM25 search index with PostgreSQL persistence.

    Implements the BM25 ranking algorithm:
    BM25(D, Q) = Sum over qi in Q of:
        IDF(qi) * (f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * |D|/avgdl))

    Where:
    - qi = query term
    - f(qi,D) = term frequency of qi in document D
    - |D| = document length (token count)
    - avgdl = average document length in corpus
    - k1, b = tuning parameters
    - IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
    """

    def __init__(
        self,
        k1: Optional[float] = None,
        b: Optional[float] = None,
        tokenizer: Optional[Tokenizer] = None,
    ):
        """
        Initialize BM25 index.

        Args:
            k1: Term frequency saturation (default from settings)
            b: Document length normalization (default from settings)
            tokenizer: Custom tokenizer (default uses settings)
        """
        search_settings = settings.search

        self.k1 = k1 if k1 is not None else search_settings.bm25_k1
        self.b = b if b is not None else search_settings.bm25_b

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = Tokenizer(
                lowercase=search_settings.lowercase,
                remove_stopwords=search_settings.remove_stopwords,
                min_token_length=search_settings.min_token_length,
            )

    async def index_document(self, chunked_doc: ChunkedDocument) -> int:
        """
        Index a chunked document.

        Args:
            chunked_doc: ChunkedDocument from chunking service

        Returns:
            Number of chunks indexed
        """
        async with async_session_factory() as session:
            # Determine which chunks to index
            chunks_to_index = chunked_doc.leaf_chunks
            if settings.search.index_parents:
                chunks_to_index = chunked_doc.chunks

            indexed_count = 0
            for chunk in chunks_to_index:
                await self._index_chunk(session, chunk, chunked_doc.doc_id)
                indexed_count += 1

            # Update corpus statistics
            await self._update_statistics(session)

            await session.commit()
            return indexed_count

    async def index_documents(self, documents: List[ChunkedDocument]) -> int:
        """
        Batch index multiple documents.

        Args:
            documents: List of ChunkedDocument objects

        Returns:
            Total number of chunks indexed
        """
        total_indexed = 0
        for doc in documents:
            count = await self.index_document(doc)
            total_indexed += count
        return total_indexed

    async def _index_chunk(
        self,
        session: AsyncSession,
        chunk: Chunk,
        doc_id: str,
    ) -> None:
        """
        Index a single chunk.

        Creates or updates both DocumentChunk (source of truth) and
        BM25IndexedChunk (search index) entries.

        Args:
            session: Database session
            chunk: Chunk to index
            doc_id: Document identifier (UUID as string)
        """
        # Tokenize and get term frequencies
        term_freqs = self.tokenizer.get_term_frequencies(chunk.text)
        token_count = sum(term_freqs.values())

        if token_count == 0:
            return

        # Convert doc_id to UUID
        doc_uuid = UUID(doc_id) if isinstance(doc_id, str) else doc_id

        # First, create or update DocumentChunk (source of truth)
        doc_chunk_stmt = insert(DocumentChunk).values(
            chunk_id=chunk.id,
            document_id=doc_uuid,
            text=chunk.text,
            char_count=len(chunk.text),
            level=chunk.level,
            page_number=chunk.metadata.get("page_number"),
            chunk_index=chunk.metadata.get("chunk_index", 0),
            position=chunk.metadata.get("position"),
            metadata={
                "parent_id": chunk.parent_id,
            },
        )
        doc_chunk_stmt = doc_chunk_stmt.on_conflict_do_update(
            index_elements=["chunk_id"],
            set_={
                "text": doc_chunk_stmt.excluded.text,
                "char_count": doc_chunk_stmt.excluded.char_count,
                "level": doc_chunk_stmt.excluded.level,
                "page_number": doc_chunk_stmt.excluded.page_number,
                "chunk_index": doc_chunk_stmt.excluded.chunk_index,
                "position": doc_chunk_stmt.excluded.position,
                "metadata": doc_chunk_stmt.excluded.metadata,
            },
        )
        await session.execute(doc_chunk_stmt)
        await session.flush()

        # Get the DocumentChunk's database ID
        doc_chunk_result = await session.execute(
            select(DocumentChunk.id).where(DocumentChunk.chunk_id == chunk.id)
        )
        document_chunk_id = doc_chunk_result.scalar_one()

        # Create or update BM25IndexedChunk (search index)
        bm25_stmt = insert(BM25IndexedChunk).values(
            document_chunk_id=document_chunk_id,
            chunk_id=chunk.id,
            doc_id=doc_uuid,
            token_count=token_count,
        )
        bm25_stmt = bm25_stmt.on_conflict_do_update(
            index_elements=["document_chunk_id"],
            set_={
                "token_count": bm25_stmt.excluded.token_count,
            },
        )
        await session.execute(bm25_stmt)
        await session.flush()

        # Get the BM25IndexedChunk's database ID
        bm25_chunk_result = await session.execute(
            select(BM25IndexedChunk.id).where(
                BM25IndexedChunk.document_chunk_id == document_chunk_id
            )
        )
        chunk_db_id = bm25_chunk_result.scalar_one()

        # Delete existing postings for this chunk (for re-indexing)
        await session.execute(
            delete(BM25Posting).where(BM25Posting.chunk_db_id == chunk_db_id)
        )

        # Process each term
        for term, freq in term_freqs.items():
            # Get or create vocabulary entry
            term_id = await self._get_or_create_term(session, term)

            # Create posting
            posting = BM25Posting(
                term_id=term_id,
                chunk_db_id=chunk_db_id,
                term_frequency=freq,
            )
            session.add(posting)

        # Update document frequencies for all terms
        await self._update_document_frequencies(session)

    async def _get_or_create_term(self, session: AsyncSession, term: str) -> int:
        """Get term ID, creating vocabulary entry if needed."""
        # Try to get existing term
        result = await session.execute(
            select(BM25Vocabulary.id).where(BM25Vocabulary.term == term)
        )
        term_id = result.scalar_one_or_none()

        if term_id is None:
            # Create new term
            vocab_entry = BM25Vocabulary(term=term, document_frequency=0)
            session.add(vocab_entry)
            await session.flush()
            term_id = vocab_entry.id

        return term_id

    async def _update_document_frequencies(self, session: AsyncSession) -> None:
        """Update document frequency for all terms."""
        # Calculate DF as count of distinct chunks per term
        subquery = (
            select(
                BM25Posting.term_id,
                func.count(func.distinct(BM25Posting.chunk_db_id)).label("df"),
            )
            .group_by(BM25Posting.term_id)
            .subquery()
        )

        # Update vocabulary with new DFs
        await session.execute(
            update(BM25Vocabulary)
            .where(BM25Vocabulary.id == subquery.c.term_id)
            .values(document_frequency=subquery.c.df)
        )

    async def _update_statistics(self, session: AsyncSession) -> None:
        """Update corpus-wide statistics."""
        # Calculate total documents and average length
        result = await session.execute(
            select(
                func.count(BM25IndexedChunk.id),
                func.coalesce(func.avg(BM25IndexedChunk.token_count), 0.0),
                func.coalesce(func.sum(BM25IndexedChunk.token_count), 0),
            )
        )
        total_docs, avg_len, total_tokens = result.one()

        # Upsert statistics
        stmt = insert(BM25CorpusStats).values(
            id=1,
            total_documents=total_docs,
            avg_document_length=float(avg_len),
            total_tokens=total_tokens,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["id"],
            set_={
                "total_documents": stmt.excluded.total_documents,
                "avg_document_length": stmt.excluded.avg_document_length,
                "total_tokens": stmt.excluded.total_tokens,
            },
        )
        await session.execute(stmt)

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_doc_ids: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Search the index and return ranked results.

        Args:
            query: Search query string
            top_k: Number of results to return (default from settings)
            filter_doc_ids: Optional list of doc_ids to restrict search to

        Returns:
            List of SearchResult objects sorted by score (descending)
        """
        if top_k is None:
            top_k = settings.search.default_top_k

        # Tokenize query
        query_terms = self.tokenizer.tokenize(query)
        if not query_terms:
            return []

        async with async_session_factory() as session:
            # Get corpus statistics
            stats = await self._get_statistics(session)
            if stats.total_documents == 0:
                return []

            # Get candidate chunks (those containing at least one query term)
            candidates = await self._get_candidate_chunks(
                session, query_terms, filter_doc_ids
            )

            if not candidates:
                return []

            # Calculate BM25 scores
            scores = []
            for chunk_data in candidates:
                score = await self._calculate_bm25_score(
                    session, query_terms, chunk_data, stats
                )
                if score > 0:
                    scores.append((chunk_data, score))

            # Sort by score descending
            scores.sort(key=lambda x: x[1], reverse=True)

            # Build results with normalized scores
            max_score = scores[0][1] if scores else 1.0
            results = []

            for chunk_data, score in scores[:top_k]:
                results.append(
                    SearchResult(
                        chunk_id=chunk_data["chunk_id"],
                        doc_id=chunk_data["doc_id"],
                        text=chunk_data["text"],
                        score=score,
                        bm25_score=score / max_score if max_score > 0 else 0.0,
                        metadata=chunk_data["metadata"],
                    )
                )

            return results

    async def _get_statistics(self, session: AsyncSession) -> BM25CorpusStats:
        """Get corpus statistics."""
        result = await session.execute(
            select(BM25CorpusStats).where(BM25CorpusStats.id == 1)
        )
        stats = result.scalar_one_or_none()

        if stats is None:
            # Return empty stats
            return BM25CorpusStats(
                id=1,
                total_documents=0,
                avg_document_length=0.0,
                total_tokens=0,
            )

        return stats

    async def _get_candidate_chunks(
        self,
        session: AsyncSession,
        query_terms: List[str],
        filter_doc_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get chunks containing at least one query term."""
        # Get term IDs for query terms
        term_result = await session.execute(
            select(BM25Vocabulary.id, BM25Vocabulary.term).where(
                BM25Vocabulary.term.in_(query_terms)
            )
        )
        term_map = {row.term: row.id for row in term_result}

        if not term_map:
            return []

        # Get chunk IDs that have postings for these terms
        posting_query = (
            select(BM25Posting.chunk_db_id)
            .where(BM25Posting.term_id.in_(term_map.values()))
            .distinct()
        )

        # Get chunk details by joining BM25IndexedChunk with DocumentChunk
        chunk_query = (
            select(
                BM25IndexedChunk.id,
                BM25IndexedChunk.chunk_id,
                BM25IndexedChunk.doc_id,
                BM25IndexedChunk.token_count,
                DocumentChunk.text,
                DocumentChunk.level,
                DocumentChunk.page_number,
                DocumentChunk.chunk_index,
                DocumentChunk.metadata,
            )
            .join(
                DocumentChunk,
                BM25IndexedChunk.document_chunk_id == DocumentChunk.id,
            )
            .where(BM25IndexedChunk.id.in_(posting_query))
        )

        # Apply document filter if provided
        if filter_doc_ids:
            # Convert string doc_ids to UUIDs if needed
            uuid_filter = [
                UUID(d) if isinstance(d, str) else d for d in filter_doc_ids
            ]
            chunk_query = chunk_query.where(
                BM25IndexedChunk.doc_id.in_(uuid_filter)
            )

        result = await session.execute(chunk_query)

        return [
            {
                "db_id": row.id,
                "chunk_id": row.chunk_id,
                "doc_id": str(row.doc_id),  # Convert UUID to string for API
                "text": row.text,
                "token_count": row.token_count,
                "metadata": {
                    "level": row.level,
                    "page_number": row.page_number,
                    "chunk_index": row.chunk_index,
                    **(row.metadata or {}),
                },
            }
            for row in result
        ]

    async def _calculate_bm25_score(
        self,
        session: AsyncSession,
        query_terms: List[str],
        chunk_data: Dict[str, Any],
        stats: BM25CorpusStats,
    ) -> float:
        """
        Calculate BM25 score for a chunk given query terms.

        BM25 = Sum over qi of:
            IDF(qi) * (f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * |D|/avgdl))
        """
        doc_length = chunk_data["token_count"]
        avgdl = stats.avg_document_length or 1.0
        N = stats.total_documents

        score = 0.0

        for term in query_terms:
            # Get term info
            term_result = await session.execute(
                select(BM25Vocabulary.id, BM25Vocabulary.document_frequency).where(
                    BM25Vocabulary.term == term
                )
            )
            term_row = term_result.one_or_none()

            if term_row is None:
                continue

            term_id, df = term_row

            # Get term frequency in this chunk
            tf_result = await session.execute(
                select(BM25Posting.term_frequency).where(
                    BM25Posting.term_id == term_id,
                    BM25Posting.chunk_db_id == chunk_data["db_id"],
                )
            )
            tf = tf_result.scalar_one_or_none() or 0

            if tf == 0:
                continue

            # Calculate IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

            # Calculate BM25 term contribution
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / avgdl)

            score += idf * (numerator / denominator)

        return score

    async def remove_document(self, doc_id: str) -> int:
        """
        Remove all chunks for a document from the index.

        Deletes DocumentChunk entries which cascades to BM25IndexedChunk
        and BM25Posting via foreign key constraints.

        Args:
            doc_id: Document identifier (UUID as string)

        Returns:
            Number of chunks removed
        """
        async with async_session_factory() as session:
            # Convert to UUID
            doc_uuid = UUID(doc_id) if isinstance(doc_id, str) else doc_id

            # Get chunks to remove (count for return value)
            result = await session.execute(
                select(DocumentChunk.id).where(DocumentChunk.document_id == doc_uuid)
            )
            chunk_ids = [row[0] for row in result]

            if not chunk_ids:
                return 0

            # Delete DocumentChunks (cascades to BM25IndexedChunk and BM25Posting)
            await session.execute(
                delete(DocumentChunk).where(DocumentChunk.document_id == doc_uuid)
            )

            # Update document frequencies
            await self._update_document_frequencies(session)

            # Clean up orphaned vocabulary entries
            await self._cleanup_vocabulary(session)

            # Update statistics
            await self._update_statistics(session)

            await session.commit()
            return len(chunk_ids)

    async def _cleanup_vocabulary(self, session: AsyncSession) -> None:
        """Remove vocabulary entries with no postings."""
        # Find terms with no postings
        subquery = select(BM25Posting.term_id).distinct().subquery()
        await session.execute(
            delete(BM25Vocabulary).where(BM25Vocabulary.id.notin_(select(subquery)))
        )

    async def clear(self) -> None:
        """Clear the entire index and document chunks."""
        async with async_session_factory() as session:
            # Order matters due to foreign key constraints
            await session.execute(delete(BM25Posting))
            await session.execute(delete(BM25IndexedChunk))
            await session.execute(delete(DocumentChunk))
            await session.execute(delete(BM25Vocabulary))
            await session.execute(delete(BM25CorpusStats))
            await session.commit()

    async def get_statistics(self) -> Dict[str, Any]:
        """Get corpus statistics."""
        async with async_session_factory() as session:
            stats = await self._get_statistics(session)
            return {
                "total_documents": stats.total_documents,
                "avg_document_length": stats.avg_document_length,
                "total_tokens": stats.total_tokens,
            }

    async def document_count(self) -> int:
        """Get number of indexed chunks."""
        async with async_session_factory() as session:
            result = await session.execute(
                select(func.count(BM25IndexedChunk.id))
            )
            return result.scalar_one()


class SearchIndexer:
    """
    High-level search indexer service.

    Wraps BM25Index and provides a simplified interface for indexing
    and searching chunked documents.
    """

    def __init__(self):
        """Initialize the search indexer with settings from config."""
        self.index = BM25Index()

    async def index_chunked_document(self, doc: ChunkedDocument) -> int:
        """
        Index a chunked document.

        Args:
            doc: ChunkedDocument from chunking service

        Returns:
            Number of chunks indexed
        """
        return await self.index.index_document(doc)

    async def index_documents(self, documents: List[ChunkedDocument]) -> int:
        """
        Batch index multiple documents.

        Args:
            documents: List of ChunkedDocument objects

        Returns:
            Total number of chunks indexed
        """
        return await self.index.index_documents(documents)

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_doc_ids: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Search indexed documents.

        Args:
            query: Search query string
            top_k: Number of results to return
            filter_doc_ids: Optional list of doc_ids to restrict search to

        Returns:
            List of SearchResult objects sorted by score
        """
        return await self.index.search(query, top_k, filter_doc_ids)

    async def remove_document(self, doc_id: str) -> int:
        """
        Remove a document from the index.

        Args:
            doc_id: Document identifier

        Returns:
            Number of chunks removed
        """
        return await self.index.remove_document(doc_id)

    async def clear(self) -> None:
        """Clear the entire index."""
        await self.index.clear()

    async def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics."""
        return await self.index.get_statistics()

    async def document_count(self) -> int:
        """Get number of indexed chunks."""
        return await self.index.document_count()


# Singleton instance
search_indexer = SearchIndexer()
