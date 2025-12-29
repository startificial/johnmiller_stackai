"""
Text Extraction Service

Handles extraction of text content from various document formats.
Currently supports PDF files using PyMuPDF (fitz).

Text Extraction Considerations:
- PyMuPDF provides fast text extraction with layout preservation
- Using get_text("blocks") preserves document structure (paragraphs, columns)
- Page-by-page processing enables progress tracking and memory efficiency
- Text cleaning normalizes whitespace
"""

import hashlib
import re
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import fitz  # PyMuPDF
from sqlalchemy import select

from backend.app.db.base import async_session_factory
from backend.app.db.models import Document
from backend.app.settings import settings


class FileTooLargeError(Exception):
    """Raised when a file exceeds the maximum allowed size."""

    pass


class UnsupportedFileTypeError(Exception):
    """Raised when a file type is not supported."""

    pass


@dataclass
class ExtractedPage:
    """Container for extracted page content."""
    page_number: int
    text: str
    char_count: int


@dataclass
class ExtractedDocument:
    """Container for extracted document content."""
    pages: list[ExtractedPage]
    full_text: str
    page_count: int
    metadata: dict
    doc_id: Optional[str] = None
    filename: Optional[str] = None
    file_hash: Optional[str] = None


class TextExtractor:
    """Handles text extraction from documents."""

    def __init__(self):
        pass

    def extract_from_pdf(self, file_path: str | Path) -> ExtractedDocument:
        """
        Extract text from a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            ExtractedDocument with pages and full text

        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            FileTooLargeError: If the file exceeds maximum size
            UnsupportedFileTypeError: If the file extension is not supported
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        extraction_settings = settings.text_extraction

        # Validate file extension
        if file_path.suffix.lower() not in extraction_settings.supported_extensions:
            raise UnsupportedFileTypeError(
                f"Unsupported file type: {file_path.suffix}. "
                f"Supported: {extraction_settings.supported_extensions}"
            )

        # Validate file size
        file_size = file_path.stat().st_size
        if file_size > extraction_settings.max_file_size:
            raise FileTooLargeError(
                f"File size ({file_size} bytes) exceeds maximum "
                f"({extraction_settings.max_file_size} bytes)"
            )

        doc = fitz.open(file_path)
        try:
            pages = []
            all_text_parts = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                raw_text = self._extract_text_from_page(page)
                cleaned_text = self._clean_text(raw_text)

                if cleaned_text:
                    pages.append(ExtractedPage(
                        page_number=page_num + 1,  # 1-indexed
                        text=cleaned_text,
                        char_count=len(cleaned_text)
                    ))
                    all_text_parts.append(cleaned_text)

            metadata = (
                self._extract_metadata(doc)
                if extraction_settings.extract_metadata
                else {}
            )

            return ExtractedDocument(
                pages=pages,
                full_text="\n\n".join(all_text_parts),
                page_count=len(doc),
                metadata=metadata
            )

        finally:
            doc.close()

    def _extract_text_from_page(self, page: fitz.Page) -> str:
        """
        Extract text from a single PDF page using block-based extraction.

        Uses 'blocks' mode to preserve paragraph structure and handle
        multi-column layouts better than simple text extraction.
        """
        blocks = page.get_text("blocks")
        text_blocks = []

        for block in blocks:
            if block[6] == 0:  # Text block (not image)
                text = block[4].strip()
                if text:
                    text_blocks.append(text)

        return "\n\n".join(text_blocks)

    def _clean_text(self, text: str) -> str:
        """Clean extracted text by removing extra whitespace and artifacts."""
        # Normalize multiple spaces to single space
        text = re.sub(r' +', ' ', text)
        # Limit multiple newlines to maximum of 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Strip whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        return text.strip()

    def _extract_metadata(self, doc: fitz.Document) -> dict:
        """Extract metadata from PDF document."""
        metadata = doc.metadata
        return {
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": metadata.get("subject", ""),
            "keywords": metadata.get("keywords", ""),
            "creator": metadata.get("creator", ""),
            "producer": metadata.get("producer", ""),
            "creation_date": metadata.get("creationDate", ""),
            "modification_date": metadata.get("modDate", ""),
        }

    def get_pdf_metadata(self, file_path: str | Path) -> dict:
        """Extract only metadata from PDF file without full text extraction."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        doc = fitz.open(file_path)
        try:
            metadata = self._extract_metadata(doc)
            metadata["page_count"] = len(doc)
            return metadata
        finally:
            doc.close()


class TextExtractionService:
    """
    High-level text extraction service with database persistence.

    Wraps TextExtractor and adds async database operations for Document records.
    """

    def __init__(self, extractor: Optional[TextExtractor] = None):
        """Initialize the service with an optional extractor instance."""
        self._extractor = extractor or TextExtractor()

    async def extract_and_persist(
        self,
        file_path: str | Path,
        doc_id: Optional[str] = None,
    ) -> ExtractedDocument:
        """
        Extract text from a file and persist Document record to database.

        Args:
            file_path: Path to the document file
            doc_id: Optional document ID (generated if not provided)

        Returns:
            ExtractedDocument with doc_id set
        """
        file_path = Path(file_path)

        # Generate doc_id if not provided
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        # Calculate file hash for deduplication
        file_hash = self._calculate_file_hash(file_path)

        # Extract text
        extracted = self._extractor.extract_from_pdf(file_path)

        # Add doc_id and file info to the extracted document
        extracted.doc_id = doc_id
        extracted.filename = file_path.name
        extracted.file_hash = file_hash

        # Persist to database
        await self._persist_document(extracted, file_path)

        return extracted

    async def _persist_document(
        self,
        extracted: ExtractedDocument,
        file_path: Path,
    ) -> None:
        """Persist Document record to database."""
        async with async_session_factory() as session:
            doc_uuid = uuid.UUID(extracted.doc_id)

            # Check if document already exists
            existing = await session.execute(
                select(Document).where(Document.id == doc_uuid)
            )
            if existing.scalar_one_or_none():
                return  # Already exists

            doc = Document(
                id=doc_uuid,
                filename=extracted.filename,
                file_hash=extracted.file_hash,
                file_size=file_path.stat().st_size,
                mime_type="application/pdf",
                page_count=extracted.page_count,
                total_char_count=len(extracted.full_text),
                doc_metadata=extracted.metadata,
            )
            session.add(doc)
            await session.commit()

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def extract_from_pdf(self, file_path: str | Path) -> ExtractedDocument:
        """
        Extract text without persisting (synchronous).

        For cases where you only need in-memory extraction.
        """
        return self._extractor.extract_from_pdf(file_path)


# Singleton instances
text_extractor = TextExtractor()
text_extraction_service = TextExtractionService(text_extractor)

