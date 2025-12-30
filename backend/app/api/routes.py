"""
API route handlers for RAG pipeline.

Endpoints:
- POST /ingest: Upload and process PDF files
- POST /query: Query the knowledge base
- Chat session management (CRUD)
- Chat messaging with conversation history
- Chat export (JSON/Markdown)
"""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.schemas import (
    BatchIngestionResponse,
    ChatExportResponse,
    ChatMessageCreate,
    ChatMessageResponse,
    ChatSessionCreate,
    ChatSessionResponse,
    ChatSessionUpdate,
    DocumentListResponse,
    DocumentResponse,
    IngestionError,
    IngestionResponse,
    QueryRequest,
    QueryResponse,
    SourceInfo,
)
from backend.app.db.base import async_session_factory, get_session
from backend.app.db.models import (
    ChatMessage,
    ChatSession,
    Document,
    DocumentChunk,
    DocumentStatus,
)
from backend.app.services.chunking import chunking_service
from backend.app.services.embedding import embedding_service
from backend.app.services.llm_response.schemas import ConversationTurn
from backend.app.services.llm_response.service import llm_response_service
from backend.app.services.search_indexer import search_indexer
from backend.app.services.text_extraction import text_extraction_service

router = APIRouter()


# =============================================================================
# Ingestion Endpoint
# =============================================================================


@router.post("/ingest", response_model=BatchIngestionResponse)
async def ingest_documents(
    files: list[UploadFile] = File(..., description="PDF files to ingest"),
):
    """
    Upload one or more PDF files for ingestion into the knowledge base.

    Pipeline for each file:
    1. Save uploaded file to temp location
    2. Extract text (TextExtractionService)
    3. Chunk document (ChunkingService)
    4. Generate embeddings (EmbeddingService)
    5. Index for BM25 search (SearchIndexer)

    Returns:
        BatchIngestionResponse with details for each processed file
    """
    results: list[IngestionResponse] = []
    errors: list[IngestionError] = []

    for file in files:
        try:
            # Validate file type
            if not file.filename or not file.filename.lower().endswith(".pdf"):
                raise ValueError(f"File must be a PDF: {file.filename}")

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = Path(tmp.name)

            try:
                # Step 1: Extract text
                extracted_doc = await text_extraction_service.extract_and_persist(
                    file_path=tmp_path,
                )

                # Step 2: Chunk document
                chunked_doc = await chunking_service.chunk_and_persist(
                    document=extracted_doc,
                    doc_id=extracted_doc.doc_id,
                )

                # Step 3: Generate embeddings (for leaf chunks)
                embed_result = await embedding_service.embed_document(
                    chunked_doc=chunked_doc,
                    embed_parents=False,
                )

                # Step 4: Index for BM25 search
                await search_indexer.index_chunked_document(chunked_doc)

                # Step 5: Update document status to completed in database
                async with async_session_factory() as session:
                    doc = await session.get(Document, extracted_doc.doc_id)
                    if doc:
                        doc.status = DocumentStatus.COMPLETED
                        await session.commit()

                # Build success response
                results.append(
                    IngestionResponse(
                        document_id=str(extracted_doc.doc_id),
                        filename=file.filename or "unknown.pdf",
                        status="completed",
                        page_count=extracted_doc.page_count,
                        chunks_created=len(chunked_doc.chunks),
                        chunks_embedded=embed_result.get("chunks_embedded", 0),
                    )
                )

            finally:
                # Clean up temp file
                tmp_path.unlink(missing_ok=True)

        except Exception as e:
            errors.append(
                IngestionError(
                    filename=file.filename or "unknown",
                    error=str(e),
                )
            )

    return BatchIngestionResponse(
        documents=results,
        errors=errors,
        total_files=len(files),
        successful=len(results),
        failed=len(errors),
    )


# =============================================================================
# Document List Endpoint
# =============================================================================


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    skip: int = Query(0, ge=0, description="Number of documents to skip"),
    limit: int = Query(20, ge=1, le=100, description="Max documents to return"),
    status: str = Query(None, description="Filter by status (e.g., 'completed')"),
    session: AsyncSession = Depends(get_session),
):
    """
    List all documents in the knowledge base with pagination.

    Returns documents ordered by creation date (newest first).
    """
    # Build base query
    base_query = select(Document)

    # Apply status filter if provided
    if status:
        base_query = base_query.where(Document.status == status)

    # Get total count
    count_query = select(func.count()).select_from(base_query.subquery())
    total_result = await session.execute(count_query)
    total = total_result.scalar() or 0

    # Get paginated documents with chunk count
    stmt = (
        select(
            Document,
            func.count(DocumentChunk.id).label("chunk_count"),
        )
        .outerjoin(DocumentChunk, Document.id == DocumentChunk.document_id)
        .group_by(Document.id)
        .order_by(Document.created_at.desc())
        .offset(skip)
        .limit(limit)
    )

    if status:
        stmt = stmt.where(Document.status == status)

    result = await session.execute(stmt)
    rows = result.all()

    documents = [
        DocumentResponse(
            document_id=str(doc.id),
            filename=doc.filename,
            status=doc.status,
            page_count=doc.page_count or 0,
            chunk_count=chunk_count,
            created_at=doc.created_at,
        )
        for doc, chunk_count in rows
    ]

    return DocumentListResponse(
        documents=documents,
        total=total,
        skip=skip,
        limit=limit,
    )


# =============================================================================
# Query Endpoint
# =============================================================================


@router.post("/query", response_model=QueryResponse)
async def query_knowledge_base(
    request: QueryRequest,
):
    """
    Query the knowledge base with a user question.

    Pipeline:
    1. Classify intent (IntentClassifier)
    2. Transform query (QueryTransformer) - for RAG-Fusion
    3. Retrieve context (HybridRetriever)
    4. Generate response (LLMResponseService)

    Returns:
        QueryResponse with intent-specific structured response
    """
    try:
        # LLMResponseService orchestrates the full pipeline
        result = await llm_response_service.generate_response(
            query=request.query,
            filter_doc_ids=request.filter_doc_ids,
        )

        # Extract sources from the result
        sources = []
        for source in result.sources_used:
            sources.append(
                SourceInfo(
                    source_id=source.source_id,
                    chunk_id=source.chunk_id,
                    document_id=source.document_id,
                    title=source.title,
                    text_excerpt=source.text_excerpt,
                    relevance_score=source.relevance_score,
                    page_number=source.page_number,
                )
            )

        return QueryResponse(
            query=result.query,
            intent=result.intent.value,
            intent_confidence=result.intent_confidence,
            response=result.response.model_dump(),
            sources=sources,
            model_used=result.model_used,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Chat Session Endpoints
# =============================================================================


@router.post("/chat/sessions", response_model=ChatSessionResponse)
async def create_chat_session(
    request: ChatSessionCreate,
    session: AsyncSession = Depends(get_session),
):
    """Create a new chat session with optional title and document filter."""
    chat_session = ChatSession(
        title=request.title,
        filter_doc_ids=request.filter_doc_ids,
    )
    session.add(chat_session)
    await session.commit()
    await session.refresh(chat_session)

    return ChatSessionResponse(
        id=str(chat_session.id),
        title=chat_session.title,
        filter_doc_ids=chat_session.filter_doc_ids,
        created_at=chat_session.created_at,
        updated_at=chat_session.updated_at,
        message_count=0,
    )


@router.get("/chat/sessions", response_model=list[ChatSessionResponse])
async def list_chat_sessions(
    skip: int = Query(0, ge=0, description="Number of sessions to skip"),
    limit: int = Query(20, ge=1, le=100, description="Max sessions to return"),
    session: AsyncSession = Depends(get_session),
):
    """List all chat sessions with pagination, ordered by most recent."""
    # Query sessions with message count
    stmt = (
        select(
            ChatSession,
            func.count(ChatMessage.id).label("message_count"),
        )
        .outerjoin(ChatMessage, ChatSession.id == ChatMessage.session_id)
        .group_by(ChatSession.id)
        .order_by(ChatSession.updated_at.desc())
        .offset(skip)
        .limit(limit)
    )

    result = await session.execute(stmt)
    rows = result.all()

    return [
        ChatSessionResponse(
            id=str(chat_session.id),
            title=chat_session.title,
            filter_doc_ids=chat_session.filter_doc_ids,
            created_at=chat_session.created_at,
            updated_at=chat_session.updated_at,
            message_count=message_count,
        )
        for chat_session, message_count in rows
    ]


@router.get("/chat/sessions/{session_id}", response_model=ChatSessionResponse)
async def get_chat_session(
    session_id: UUID,
    session: AsyncSession = Depends(get_session),
):
    """Get a single chat session by ID."""
    stmt = (
        select(
            ChatSession,
            func.count(ChatMessage.id).label("message_count"),
        )
        .outerjoin(ChatMessage, ChatSession.id == ChatMessage.session_id)
        .where(ChatSession.id == session_id)
        .group_by(ChatSession.id)
    )

    result = await session.execute(stmt)
    row = result.first()

    if not row:
        raise HTTPException(status_code=404, detail="Chat session not found")

    chat_session, message_count = row
    return ChatSessionResponse(
        id=str(chat_session.id),
        title=chat_session.title,
        filter_doc_ids=chat_session.filter_doc_ids,
        created_at=chat_session.created_at,
        updated_at=chat_session.updated_at,
        message_count=message_count,
    )


@router.patch("/chat/sessions/{session_id}", response_model=ChatSessionResponse)
async def update_chat_session(
    session_id: UUID,
    request: ChatSessionUpdate,
    session: AsyncSession = Depends(get_session),
):
    """Update a chat session's title or document filter."""
    chat_session = await session.get(ChatSession, session_id)
    if not chat_session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    # Update fields if provided
    if request.title is not None:
        chat_session.title = request.title
    if request.filter_doc_ids is not None:
        chat_session.filter_doc_ids = request.filter_doc_ids

    await session.commit()
    await session.refresh(chat_session)

    # Get message count
    count_stmt = select(func.count(ChatMessage.id)).where(
        ChatMessage.session_id == session_id
    )
    count_result = await session.execute(count_stmt)
    message_count = count_result.scalar() or 0

    return ChatSessionResponse(
        id=str(chat_session.id),
        title=chat_session.title,
        filter_doc_ids=chat_session.filter_doc_ids,
        created_at=chat_session.created_at,
        updated_at=chat_session.updated_at,
        message_count=message_count,
    )


@router.delete("/chat/sessions/{session_id}")
async def delete_chat_session(
    session_id: UUID,
    session: AsyncSession = Depends(get_session),
):
    """Delete a chat session and all its messages."""
    chat_session = await session.get(ChatSession, session_id)
    if not chat_session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    await session.delete(chat_session)
    await session.commit()

    return {"message": "Chat session deleted successfully"}


# =============================================================================
# Chat Message Endpoints
# =============================================================================


@router.post(
    "/chat/sessions/{session_id}/messages", response_model=ChatMessageResponse
)
async def send_chat_message(
    session_id: UUID,
    message: ChatMessageCreate,
    session: AsyncSession = Depends(get_session),
):
    """
    Send a message in a chat session and get an assistant response.

    Pipeline:
    1. Load conversation history from DB
    2. Get session's document filter
    3. Call LLMResponseService with history + filter
    4. Store both user message and assistant response
    5. Return assistant response
    """
    # Get chat session
    chat_session = await session.get(ChatSession, session_id)
    if not chat_session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    # Load conversation history
    history_stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
    )
    history_result = await session.execute(history_stmt)
    history_messages = history_result.scalars().all()

    # Convert to ConversationTurn format
    conversation_history = [
        ConversationTurn(role=msg.role, content=msg.content)
        for msg in history_messages
    ]

    # Store user message
    user_message = ChatMessage(
        session_id=session_id,
        role="user",
        content=message.content,
    )
    session.add(user_message)

    try:
        # Generate response using LLM service
        result = await llm_response_service.generate_response(
            query=message.content,
            conversation_history=conversation_history,
            filter_doc_ids=chat_session.filter_doc_ids,
        )

        # Extract sources for storage
        sources_data = [
            {
                "source_id": s.source_id,
                "chunk_id": s.chunk_id,
                "document_id": s.document_id,
                "title": s.title,
                "text_excerpt": s.text_excerpt,
                "relevance_score": s.relevance_score,
                "page_number": s.page_number,
            }
            for s in result.sources_used
        ]

        # Store assistant response
        assistant_message = ChatMessage(
            session_id=session_id,
            role="assistant",
            content=result.response.model_dump_json(),
            intent=result.intent.value,
            response_metadata={
                "sources": sources_data,
                "intent_confidence": result.intent_confidence,
                "model_used": result.model_used,
            },
        )
        session.add(assistant_message)
        await session.commit()
        await session.refresh(assistant_message)

        # Build response
        return ChatMessageResponse(
            id=assistant_message.id,
            role="assistant",
            content=assistant_message.content,
            intent=assistant_message.intent,
            sources=[
                SourceInfo(**s) for s in sources_data
            ] if sources_data else None,
            created_at=assistant_message.created_at,
        )

    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/chat/sessions/{session_id}/messages", response_model=list[ChatMessageResponse]
)
async def get_chat_history(
    session_id: UUID,
    skip: int = Query(0, ge=0, description="Number of messages to skip"),
    limit: int = Query(50, ge=1, le=200, description="Max messages to return"),
    session: AsyncSession = Depends(get_session),
):
    """Get chat history for a session with pagination."""
    # Verify session exists
    chat_session = await session.get(ChatSession, session_id)
    if not chat_session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
        .offset(skip)
        .limit(limit)
    )

    result = await session.execute(stmt)
    messages = result.scalars().all()

    return [
        ChatMessageResponse(
            id=msg.id,
            role=msg.role,
            content=msg.content,
            intent=msg.intent,
            sources=_extract_sources(msg.response_metadata) if msg.response_metadata else None,
            created_at=msg.created_at,
        )
        for msg in messages
    ]


def _extract_sources(metadata: Optional[dict]) -> Optional[list[SourceInfo]]:
    """Extract sources from response metadata."""
    if not metadata or "sources" not in metadata:
        return None
    return [SourceInfo(**s) for s in metadata["sources"]]


# =============================================================================
# Chat Export Endpoint
# =============================================================================


@router.get("/chat/sessions/{session_id}/export")
async def export_chat_session(
    session_id: UUID,
    format: str = Query("json", regex="^(json|markdown)$", description="Export format"),
    session: AsyncSession = Depends(get_session),
):
    """Export chat session history as JSON or Markdown."""
    # Get session with messages
    chat_session = await session.get(ChatSession, session_id)
    if not chat_session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
    )
    result = await session.execute(stmt)
    messages = result.scalars().all()

    message_responses = [
        ChatMessageResponse(
            id=msg.id,
            role=msg.role,
            content=msg.content,
            intent=msg.intent,
            sources=_extract_sources(msg.response_metadata) if msg.response_metadata else None,
            created_at=msg.created_at,
        )
        for msg in messages
    ]

    if format == "markdown":
        return _export_as_markdown(chat_session, message_responses)

    # Default: JSON format
    return ChatExportResponse(
        session_id=str(chat_session.id),
        title=chat_session.title,
        filter_doc_ids=chat_session.filter_doc_ids,
        exported_at=datetime.utcnow(),
        message_count=len(messages),
        messages=message_responses,
    )


def _export_as_markdown(
    chat_session: ChatSession, messages: list[ChatMessageResponse]
) -> PlainTextResponse:
    """Generate markdown export of chat session."""
    lines = []

    # Header
    title = chat_session.title or "Untitled Chat Session"
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Session ID:** {chat_session.id}")
    lines.append(f"**Created:** {chat_session.created_at.isoformat()}")
    if chat_session.filter_doc_ids:
        lines.append(f"**Document Filter:** {', '.join(chat_session.filter_doc_ids)}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Messages
    for msg in messages:
        role_label = "User" if msg.role == "user" else "Assistant"
        lines.append(f"## {role_label}")
        lines.append(f"*{msg.created_at.isoformat()}*")
        lines.append("")
        lines.append(msg.content)
        lines.append("")

        if msg.intent:
            lines.append(f"**Intent:** {msg.intent}")
            lines.append("")

        if msg.sources:
            lines.append("**Sources:**")
            for source in msg.sources:
                lines.append(f"- {source.title or source.chunk_id} (relevance: {source.relevance_score:.2f})")
            lines.append("")

        lines.append("---")
        lines.append("")

    content = "\n".join(lines)
    return PlainTextResponse(
        content=content,
        media_type="text/markdown",
        headers={
            "Content-Disposition": f'attachment; filename="chat-{chat_session.id}.md"'
        },
    )
