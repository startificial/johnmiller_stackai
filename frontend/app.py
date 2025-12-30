"""
Main Streamlit application for RAG Knowledge Base.

Run with: streamlit run frontend/app.py
"""

import os
import sys

# Add frontend directory to path for imports
frontend_dir = os.path.dirname(os.path.abspath(__file__))
if frontend_dir not in sys.path:
    sys.path.insert(0, frontend_dir)

import streamlit as st

from api.client import APIClient, APIError
from api.models import ChatMessageResponse, DocumentResponse, IngestionResponse
from components.chat import render_chat_history, render_chat_input
from components.document_viewer import render_document_list, render_document_summary
from components.file_upload import render_upload_section
from components.session_sidebar import (
    handle_session_action,
    render_session_header,
    render_session_sidebar,
)
from config import config


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title=config.page_title,
    page_icon=config.page_icon,
    layout=config.layout,
    initial_sidebar_state="expanded",
)


# =============================================================================
# Session State Initialization
# =============================================================================


def init_session_state():
    """Initialize all session state variables."""
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_documents" not in st.session_state:
        st.session_state.uploaded_documents = []
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False


init_session_state()


# =============================================================================
# API Client
# =============================================================================

api_client = APIClient(config.api_base_url, timeout=config.request_timeout)


# =============================================================================
# Document Loading
# =============================================================================


def load_documents_from_backend():
    """Load existing documents from backend on first load."""
    if st.session_state.documents_loaded:
        return

    try:
        # Load all documents (status filter removed - documents may remain in pending)
        response = api_client.list_documents()
        # Convert DocumentResponse to IngestionResponse format for compatibility
        st.session_state.uploaded_documents = [
            IngestionResponse(
                document_id=doc.document_id,
                filename=doc.filename,
                status=doc.status,
                page_count=doc.page_count,
                chunks_created=doc.chunk_count,
                chunks_embedded=doc.chunk_count,  # Approximate
            )
            for doc in response.documents
        ]
        st.session_state.documents_loaded = True
    except APIError:
        # Silently fail - documents will show as empty until upload
        st.session_state.documents_loaded = True


# =============================================================================
# Sidebar
# =============================================================================


def render_sidebar():
    """Render the sidebar with upload and session management."""
    # Load existing documents from backend on first load
    load_documents_from_backend()

    with st.sidebar:
        st.title("Knowledge Base Chat")

        # Upload section
        with st.expander("Upload Documents", expanded=False):
            result = render_upload_section(api_client)
            if result and result.documents:
                # Add to uploaded documents list
                st.session_state.uploaded_documents.extend(result.documents)

        # Document summary
        if st.session_state.uploaded_documents:
            render_document_summary(st.session_state.uploaded_documents)

        st.divider()

        # Session management
        try:
            sessions = api_client.list_sessions()
        except APIError as e:
            st.error(f"Failed to load sessions: {e.message}")
            sessions = []

        action, session_id = render_session_sidebar(
            api_client,
            sessions,
            st.session_state.current_session_id,
        )

        if action:
            new_session_id = handle_session_action(action, session_id, api_client)
            if action == "delete" and session_id == st.session_state.current_session_id:
                st.session_state.current_session_id = None
                st.session_state.messages = []
            elif new_session_id:
                st.session_state.current_session_id = new_session_id
            st.rerun()

        # Document viewer in sidebar
        st.divider()
        with st.expander("Uploaded Documents", expanded=False):
            render_document_list(st.session_state.uploaded_documents)


# =============================================================================
# Main Content
# =============================================================================


def render_main_content():
    """Render the main chat area."""
    current_session = st.session_state.current_session_id

    if not current_session:
        _render_welcome_screen()
        return

    # Get session info
    try:
        session_info = api_client.get_session(current_session)
        render_session_header(session_info)
    except APIError as e:
        st.error(f"Failed to load session: {e.message}")
        return

    st.divider()

    # Chat history container
    chat_container = st.container()
    with chat_container:
        _render_messages()

    # Chat input
    if user_input := render_chat_input():
        _handle_user_input(user_input, current_session)


def _render_welcome_screen():
    """Render welcome screen when no session is selected."""
    st.title("RAG Knowledge Base")
    st.markdown(
        """
        Welcome to the RAG Knowledge Base chat interface.

        **Get started:**
        1. Upload PDF documents using the sidebar
        2. Create a new chat session
        3. Ask questions about your documents

        **Features:**
        - Chat with your documents using AI
        - View source citations for answers
        - Multiple chat sessions for different topics
        - Export conversations
        """
    )

    # Quick start button
    if st.button("Start New Chat", type="primary"):
        try:
            new_session = api_client.create_session()
            st.session_state.current_session_id = new_session.id
            st.session_state.messages = []
            st.rerun()
        except APIError as e:
            st.error(f"Failed to create session: {e.message}")


def _render_messages():
    """Render chat messages from session state."""
    messages = st.session_state.messages

    if not messages:
        st.info("Start a conversation by typing a message below.")
        return

    # Convert dict messages to ChatMessageResponse if needed
    for msg in messages:
        if isinstance(msg, dict):
            # User message added locally
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        else:
            # ChatMessageResponse from API
            render_chat_history([msg])


def _handle_user_input(user_input: str, session_id: str):
    """Handle user message submission."""
    # Add user message to display immediately
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
    })

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = api_client.send_message(session_id, user_input)

                # Replace temp user message with actual response
                # (backend returns both user and assistant messages)
                st.session_state.messages[-1] = ChatMessageResponse(
                    id=0,
                    role="user",
                    content=user_input,
                    intent=None,
                    sources=None,
                    created_at=response.created_at,
                )
                st.session_state.messages.append(response)

                # Render the response
                from components.response_renderer import render_response
                from components.citations import render_inline_citations

                try:
                    render_response(response.content, response.intent)
                    if response.sources:
                        render_inline_citations(response.sources)
                except Exception as render_error:
                    # If rendering fails, show the raw content as fallback
                    st.warning(f"Error rendering response: {render_error}")
                    st.code(response.content, language="json")

            except APIError as e:
                st.error(f"Failed to get response: {e.message}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    st.rerun()


# =============================================================================
# Main
# =============================================================================


def main():
    """Main application entry point."""
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()
