"""
HTTP client for backend API communication.
"""

from typing import BinaryIO, Optional

import httpx

from .models import (
    BatchIngestionResponse,
    ChatExportResponse,
    ChatMessageResponse,
    ChatSessionCreate,
    ChatSessionResponse,
    ChatSessionUpdate,
    DocumentListResponse,
    QueryResponse,
)


class APIError(Exception):
    """API communication error."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class APIClient:
    """HTTP client for backend API calls."""

    def __init__(self, base_url: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _handle_response(self, response: httpx.Response) -> dict:
        """Handle HTTP response and raise appropriate errors."""
        try:
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_detail = e.response.json().get("detail", str(e))
            except Exception:
                error_detail = e.response.text or str(e)
            raise APIError(error_detail, e.response.status_code) from e
        except httpx.RequestError as e:
            raise APIError(f"Connection error: {str(e)}") from e

    # =========================================================================
    # Document Ingestion
    # =========================================================================

    def ingest_documents(
        self, files: list[tuple[str, BinaryIO]]
    ) -> BatchIngestionResponse:
        """
        Upload PDF files for ingestion.

        Args:
            files: List of (filename, file_object) tuples

        Returns:
            BatchIngestionResponse with results
        """
        with httpx.Client(timeout=self.timeout) as client:
            file_data = [("files", (name, f, "application/pdf")) for name, f in files]
            response = client.post(f"{self.base_url}/ingest", files=file_data)
            data = self._handle_response(response)
            return BatchIngestionResponse(**data)

    def list_documents(
        self,
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = None,
    ) -> DocumentListResponse:
        """
        List all documents in the knowledge base.

        Args:
            skip: Number of documents to skip
            limit: Maximum documents to return
            status: Filter by status (e.g., 'completed')

        Returns:
            DocumentListResponse with list of documents
        """
        with httpx.Client(timeout=self.timeout) as client:
            params = {"skip": skip, "limit": limit}
            if status:
                params["status"] = status
            response = client.get(f"{self.base_url}/documents", params=params)
            data = self._handle_response(response)
            return DocumentListResponse(**data)

    # =========================================================================
    # Query
    # =========================================================================

    def query(
        self, query: str, filter_doc_ids: Optional[list[str]] = None
    ) -> QueryResponse:
        """
        Query the knowledge base.

        Args:
            query: User question
            filter_doc_ids: Optional document IDs to filter results

        Returns:
            QueryResponse with intent and answer
        """
        with httpx.Client(timeout=self.timeout) as client:
            payload = {"query": query}
            if filter_doc_ids:
                payload["filter_doc_ids"] = filter_doc_ids
            response = client.post(f"{self.base_url}/query", json=payload)
            data = self._handle_response(response)
            return QueryResponse(**data)

    # =========================================================================
    # Chat Sessions
    # =========================================================================

    def create_session(
        self,
        title: Optional[str] = None,
        filter_doc_ids: Optional[list[str]] = None,
    ) -> ChatSessionResponse:
        """
        Create a new chat session.

        Args:
            title: Optional session title
            filter_doc_ids: Optional document filter

        Returns:
            ChatSessionResponse with new session details
        """
        with httpx.Client(timeout=self.timeout) as client:
            payload = ChatSessionCreate(
                title=title, filter_doc_ids=filter_doc_ids
            ).model_dump(exclude_none=True)
            response = client.post(f"{self.base_url}/chat/sessions", json=payload)
            data = self._handle_response(response)
            return ChatSessionResponse(**data)

    def list_sessions(
        self, skip: int = 0, limit: int = 20
    ) -> list[ChatSessionResponse]:
        """
        List all chat sessions.

        Args:
            skip: Number of sessions to skip
            limit: Maximum sessions to return

        Returns:
            List of ChatSessionResponse
        """
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(
                f"{self.base_url}/chat/sessions",
                params={"skip": skip, "limit": limit},
            )
            data = self._handle_response(response)
            return [ChatSessionResponse(**s) for s in data]

    def get_session(self, session_id: str) -> ChatSessionResponse:
        """
        Get a single chat session.

        Args:
            session_id: Session UUID

        Returns:
            ChatSessionResponse
        """
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(f"{self.base_url}/chat/sessions/{session_id}")
            data = self._handle_response(response)
            return ChatSessionResponse(**data)

    def update_session(
        self,
        session_id: str,
        title: Optional[str] = None,
        filter_doc_ids: Optional[list[str]] = None,
    ) -> ChatSessionResponse:
        """
        Update a chat session.

        Args:
            session_id: Session UUID
            title: New title
            filter_doc_ids: New document filter

        Returns:
            Updated ChatSessionResponse
        """
        with httpx.Client(timeout=self.timeout) as client:
            payload = ChatSessionUpdate(
                title=title, filter_doc_ids=filter_doc_ids
            ).model_dump(exclude_none=True)
            response = client.patch(
                f"{self.base_url}/chat/sessions/{session_id}", json=payload
            )
            data = self._handle_response(response)
            return ChatSessionResponse(**data)

    def delete_session(self, session_id: str) -> None:
        """
        Delete a chat session.

        Args:
            session_id: Session UUID
        """
        with httpx.Client(timeout=self.timeout) as client:
            response = client.delete(f"{self.base_url}/chat/sessions/{session_id}")
            response.raise_for_status()

    # =========================================================================
    # Chat Messages
    # =========================================================================

    def send_message(self, session_id: str, content: str) -> ChatMessageResponse:
        """
        Send a message to a chat session and get response.

        Args:
            session_id: Session UUID
            content: Message content

        Returns:
            ChatMessageResponse with assistant's reply
        """
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/chat/sessions/{session_id}/messages",
                json={"content": content},
            )
            data = self._handle_response(response)
            return ChatMessageResponse(**data)

    def get_messages(
        self, session_id: str, skip: int = 0, limit: int = 50
    ) -> list[ChatMessageResponse]:
        """
        Get messages from a chat session.

        Args:
            session_id: Session UUID
            skip: Number of messages to skip
            limit: Maximum messages to return

        Returns:
            List of ChatMessageResponse
        """
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(
                f"{self.base_url}/chat/sessions/{session_id}/messages",
                params={"skip": skip, "limit": limit},
            )
            data = self._handle_response(response)
            return [ChatMessageResponse(**m) for m in data]

    # =========================================================================
    # Export
    # =========================================================================

    def export_session(
        self, session_id: str, format: str = "json"
    ) -> ChatExportResponse | str:
        """
        Export a chat session.

        Args:
            session_id: Session UUID
            format: Export format ('json' or 'markdown')

        Returns:
            ChatExportResponse for JSON, string for markdown
        """
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(
                f"{self.base_url}/chat/sessions/{session_id}/export",
                params={"format": format},
            )
            if format == "markdown":
                response.raise_for_status()
                return response.text
            data = self._handle_response(response)
            return ChatExportResponse(**data)
