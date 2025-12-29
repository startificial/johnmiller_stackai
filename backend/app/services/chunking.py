"""
Structural Recursive Chunker

Recursive chunker that uses structural boundaries (paragraphs, sections, etc.)

This implements the "retrieve small, read big" pattern:
- Small leaf chunks for precise retrieval
- Parent chunks for broader context during LLM generation
"""

import re
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from backend.app.services.text_extraction import ExtractedDocument
from backend.app.settings import settings


@dataclass
class Chunk:
    """Represents a text chunk with hierarchical relationships."""

    id: str
    text: str
    level: int
    parent_id: Optional[str] = None
    child_ids: List[str] = None
    metadata: Dict = None
    embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.child_ids is None:
            self.child_ids = []
        if self.metadata is None:
            self.metadata = {}

    @property
    def is_parent(self) -> bool:
        """Check if this chunk has children."""
        return len(self.child_ids) > 0

    @property
    def is_leaf(self) -> bool:
        """Check if this chunk is a leaf node (no children)."""
        return len(self.child_ids) == 0

    # Convenience properties for compatibility with upload route
    @property
    def content(self) -> str:
        """Alias for text attribute."""
        return self.text

    @property
    def page_number(self) -> int:
        """Get page number from metadata."""
        return self.metadata.get("page_number", 1)

    @property
    def chunk_index(self) -> int:
        """Get chunk index from metadata or derive from id."""
        if "chunk_index" in self.metadata:
            return self.metadata["chunk_index"]
        # Extract from id (e.g., "doc_p1_L0_5" -> 5)
        try:
            return int(self.id.split("_")[-1])
        except (ValueError, IndexError):
            return 0

    @property
    def start_char(self) -> Optional[int]:
        """Get start character position from metadata."""
        return self.metadata.get("position")

    @property
    def end_char(self) -> Optional[int]:
        """Get end character position (start + length)."""
        pos = self.metadata.get("position")
        if pos is not None:
            return pos + self.metadata.get("char_count", len(self.text))
        return None


@dataclass
class ChunkedDocument:
    """Container for chunked document results."""

    chunks: List[Chunk]
    parent_chunks: List[Chunk]
    leaf_chunks: List[Chunk]
    page_count: int
    doc_id: str

    @property
    def child_chunks(self) -> List[Chunk]:
        """Alias for leaf_chunks for compatibility with upload route."""
        return self.leaf_chunks

    @classmethod
    def from_chunks(
        cls, chunks: List[Chunk], page_count: int, doc_id: str
    ) -> "ChunkedDocument":
        """Create ChunkedDocument from a list of chunks."""
        parent_chunks = [c for c in chunks if c.is_parent]
        leaf_chunks = [c for c in chunks if c.is_leaf]
        return cls(
            chunks=chunks,
            parent_chunks=parent_chunks,
            leaf_chunks=leaf_chunks,
            page_count=page_count,
            doc_id=doc_id,
        )


class StructuralChunker:
    """
    Recursive chunker that creates hierarchical chunks from extracted documents.

    Creates a two-level hierarchy:
    - Level 0 (Parent): Larger chunks for context (e.g., full pages or sections)
    - Level 1 (Leaf): Smaller chunks for precise retrieval (e.g., paragraphs)
    """

    def __init__(
        self,
        parent_chunk_size: Optional[int] = None,
        child_chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        min_chunk_size: Optional[int] = None,
    ):
        """
        Initialize the chunker with size parameters.

        Args:
            parent_chunk_size: Target size for parent chunks (characters).
                              Defaults to settings.chunking.parent_chunk_size
            child_chunk_size: Target size for leaf chunks (characters).
                             Defaults to settings.chunking.child_chunk_size
            chunk_overlap: Overlap between consecutive chunks (characters).
                          Defaults to settings.chunking.chunk_overlap
            min_chunk_size: Minimum chunk size to create (characters).
                           Defaults to settings.chunking.min_chunk_size
        """
        chunking_settings = settings.chunking
        self.parent_chunk_size = (
            parent_chunk_size
            if parent_chunk_size is not None
            else chunking_settings.parent_chunk_size
        )
        self.child_chunk_size = (
            child_chunk_size
            if child_chunk_size is not None
            else chunking_settings.child_chunk_size
        )
        self.chunk_overlap = (
            chunk_overlap
            if chunk_overlap is not None
            else chunking_settings.chunk_overlap
        )
        self.min_chunk_size = (
            min_chunk_size
            if min_chunk_size is not None
            else chunking_settings.min_chunk_size
        )

    def chunk_document(
        self, document: ExtractedDocument, doc_id: Optional[str] = None
    ) -> ChunkedDocument:
        """
        Chunk an extracted document into hierarchical chunks.

        Args:
            document: ExtractedDocument from text extraction service
            doc_id: Optional document identifier (generated if not provided)

        Returns:
            ChunkedDocument with parent and leaf chunks
        """
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        all_chunks: List[Chunk] = []
        chunk_counter = 0

        for page in document.pages:
            page_num = page.page_number
            page_text = page.text

            if len(page_text.strip()) < self.min_chunk_size:
                # Page too small, create single leaf chunk
                if page_text.strip():
                    chunk_id = f"{doc_id}_p{page_num}_L1_{chunk_counter}"
                    all_chunks.append(
                        Chunk(
                            id=chunk_id,
                            text=page_text.strip(),
                            level=1,
                            metadata={
                                "page_number": page_num,
                                "chunk_index": chunk_counter,
                                "char_count": len(page_text.strip()),
                            },
                        )
                    )
                    chunk_counter += 1
                continue

            # Create parent chunks from page
            parent_texts = self._split_into_chunks(
                page_text, self.parent_chunk_size, self.chunk_overlap
            )

            for parent_idx, parent_text in enumerate(parent_texts):
                parent_id = f"{doc_id}_p{page_num}_L0_{chunk_counter}"
                parent_chunk = Chunk(
                    id=parent_id,
                    text=parent_text,
                    level=0,
                    metadata={
                        "page_number": page_num,
                        "chunk_index": chunk_counter,
                        "char_count": len(parent_text),
                    },
                )
                chunk_counter += 1

                # Create child chunks from parent
                child_texts = self._split_into_chunks(
                    parent_text, self.child_chunk_size, self.chunk_overlap
                )

                child_ids = []
                for child_idx, child_text in enumerate(child_texts):
                    if len(child_text.strip()) < self.min_chunk_size // 2:
                        continue

                    child_id = f"{doc_id}_p{page_num}_L1_{chunk_counter}"
                    child_chunk = Chunk(
                        id=child_id,
                        text=child_text,
                        level=1,
                        parent_id=parent_id,
                        metadata={
                            "page_number": page_num,
                            "chunk_index": chunk_counter,
                            "char_count": len(child_text),
                        },
                    )
                    all_chunks.append(child_chunk)
                    child_ids.append(child_id)
                    chunk_counter += 1

                parent_chunk.child_ids = child_ids
                all_chunks.append(parent_chunk)

        return ChunkedDocument.from_chunks(
            chunks=all_chunks,
            page_count=document.page_count,
            doc_id=doc_id,
        )

    def _split_into_chunks(
        self, text: str, chunk_size: int, overlap: int
    ) -> List[str]:
        """
        Split text into chunks respecting structural boundaries.

        Attempts to split on paragraph boundaries first, then sentences,
        then falls back to character-based splitting.
        """
        if len(text) <= chunk_size:
            return [text]

        # Split into paragraphs first
        paragraphs = self._split_into_paragraphs(text)

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(para) + 2 > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Start new chunk with overlap from end of previous
                    if overlap > 0 and len(current_chunk) > overlap:
                        current_chunk = current_chunk[-overlap:] + "\n\n" + para
                    else:
                        current_chunk = para
                else:
                    # Paragraph itself is too large, split it further
                    if len(para) > chunk_size:
                        sub_chunks = self._split_large_paragraph(
                            para, chunk_size, overlap
                        )
                        chunks.extend(sub_chunks[:-1])
                        current_chunk = sub_chunks[-1] if sub_chunks else ""
                    else:
                        current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs based on double newlines."""
        # Split on double newlines (paragraph boundaries)
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_large_paragraph(
        self, text: str, chunk_size: int, overlap: int
    ) -> List[str]:
        """Split a large paragraph into smaller chunks by sentences."""
        # Try to split by sentences first
        sentences = re.split(r"(?<=[.!?])\s+", text)

        if len(sentences) > 1:
            chunks = []
            current_chunk = ""

            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 > chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        if overlap > 0 and len(current_chunk) > overlap:
                            current_chunk = current_chunk[-overlap:] + " " + sentence
                        else:
                            current_chunk = sentence
                    else:
                        # Single sentence too large, split by characters
                        chunks.extend(
                            self._split_by_characters(sentence, chunk_size, overlap)
                        )
                        current_chunk = ""
                else:
                    current_chunk += (" " if current_chunk else "") + sentence

            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            return chunks

        # Fall back to character-based splitting
        return self._split_by_characters(text, chunk_size, overlap)

    def _split_by_characters(
        self, text: str, chunk_size: int, overlap: int
    ) -> List[str]:
        """Split text by character count with overlap."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at a word boundary
            if end < len(text):
                space_idx = text.rfind(" ", start, end)
                if space_idx > start:
                    end = space_idx

            chunks.append(text[start:end].strip())
            start = end - overlap if overlap > 0 else end

        return chunks


# Singleton instance with default settings
structural_chunker = StructuralChunker()
