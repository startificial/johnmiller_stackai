"""
Pydantic schemas for API request/response models.

Ingestion Schemas:
- IngestionResponse: Response for single document ingestion
- BatchIngestionResponse: Response for batch document ingestion

Query Schemas:
- QueryRequest: Request for knowledge base query
- QueryResponse: Response with intent-specific answer

Chat Schemas:
- ChatSessionCreate/Update/Response: Session management
- ChatMessageCreate/Response: Message handling
- ChatExportResponse: Session export format
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Ingestion Schemas
# =============================================================================


class IngestionResponse(BaseModel):
    """Response for a single document ingestion."""

    document_id: str = Field(..., description="UUID of the ingested document")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Processing status")
    page_count: int = Field(..., description="Number of pages extracted")
    chunks_created: int = Field(..., description="Total chunks created")
    chunks_embedded: int = Field(..., description="Chunks with embeddings")

    model_config = {"from_attributes": True}


class IngestionError(BaseModel):
    """Error details for failed ingestion."""

    filename: str = Field(..., description="Filename that failed")
    error: str = Field(..., description="Error message")


class BatchIngestionResponse(BaseModel):
    """Response for batch document ingestion."""

    documents: list[IngestionResponse] = Field(
        default_factory=list, description="Successfully ingested documents"
    )
    errors: list[IngestionError] = Field(
        default_factory=list, description="Failed ingestions"
    )
    total_files: int = Field(..., description="Total files submitted")
    successful: int = Field(..., description="Successfully processed count")
    failed: int = Field(..., description="Failed count")


# =============================================================================
# Query Schemas
# =============================================================================


class QueryRequest(BaseModel):
    """Request for querying the knowledge base."""

    query: str = Field(..., min_length=1, description="User query")
    filter_doc_ids: Optional[list[str]] = Field(
        None, description="Restrict search to specific document IDs"
    )


class SourceInfo(BaseModel):
    """Citation source information."""

    source_id: str
    chunk_id: str
    document_id: str
    title: Optional[str] = None
    text_excerpt: str
    relevance_score: float
    page_number: Optional[int] = None


class QueryResponse(BaseModel):
    """Response for a knowledge base query."""

    query: str = Field(..., description="Original query")
    intent: str = Field(..., description="Classified intent")
    intent_confidence: float = Field(..., description="Intent confidence score")
    response: dict[str, Any] = Field(..., description="Intent-specific response")
    sources: list[SourceInfo] = Field(
        default_factory=list, description="Citation sources"
    )
    model_used: str = Field(..., description="LLM model used for generation")


# =============================================================================
# Chat Schemas
# =============================================================================


class ChatSessionCreate(BaseModel):
    """Request to create a new chat session."""

    title: Optional[str] = Field(None, max_length=255, description="Session title")
    filter_doc_ids: Optional[list[str]] = Field(
        None, description="Restrict chat to specific document IDs"
    )


class ChatSessionUpdate(BaseModel):
    """Request to update a chat session."""

    title: Optional[str] = Field(None, max_length=255, description="New session title")
    filter_doc_ids: Optional[list[str]] = Field(
        None, description="Update document filter"
    )


class ChatSessionResponse(BaseModel):
    """Response containing chat session details."""

    id: str = Field(..., description="Session UUID")
    title: Optional[str] = Field(None, description="Session title")
    filter_doc_ids: Optional[list[str]] = Field(
        None, description="Document filter list"
    )
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    message_count: int = Field(..., description="Number of messages in session")

    model_config = {"from_attributes": True}


class ChatMessageCreate(BaseModel):
    """Request to send a chat message."""

    content: str = Field(..., min_length=1, description="Message content")


class ChatMessageResponse(BaseModel):
    """Response containing a chat message."""

    id: int = Field(..., description="Message ID")
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    intent: Optional[str] = Field(None, description="Intent classification (assistant only)")
    sources: Optional[list[SourceInfo]] = Field(
        None, description="Citation sources (assistant only)"
    )
    created_at: datetime = Field(..., description="Message timestamp")

    model_config = {"from_attributes": True}


class ChatExportResponse(BaseModel):
    """Response for exporting a chat session."""

    session_id: str = Field(..., description="Session UUID")
    title: Optional[str] = Field(None, description="Session title")
    filter_doc_ids: Optional[list[str]] = Field(
        None, description="Document filter used"
    )
    exported_at: datetime = Field(..., description="Export timestamp")
    message_count: int = Field(..., description="Number of messages")
    messages: list[ChatMessageResponse] = Field(
        default_factory=list, description="All messages in session"
    )
