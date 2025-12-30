"""
Chat session management sidebar component.
"""

from typing import Optional

import streamlit as st

from api.client import APIClient, APIError
from api.models import ChatSessionResponse
from utils.helpers import format_datetime


def render_session_sidebar(
    api_client: APIClient,
    sessions: list[ChatSessionResponse],
    current_session_id: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """
    Render session management in sidebar.

    Args:
        api_client: APIClient instance
        sessions: List of existing sessions
        current_session_id: Currently selected session ID

    Returns:
        Tuple of (action, session_id) where action is:
        - "create": Create new session
        - "select": Select existing session
        - "delete": Delete session
        - None: No action
    """
    st.sidebar.title("Chat Sessions")

    # New session button
    if st.sidebar.button("+ New Chat", use_container_width=True, type="primary"):
        return ("create", None)

    st.sidebar.divider()

    # List existing sessions
    if not sessions:
        st.sidebar.info("No chat sessions yet")
        return (None, None)

    for session in sessions:
        action = _render_session_item(session, current_session_id)
        if action:
            return action

    return (None, None)


def _render_session_item(
    session: ChatSessionResponse,
    current_session_id: Optional[str],
) -> Optional[tuple[str, str]]:
    """
    Render a single session item.

    Args:
        session: ChatSessionResponse
        current_session_id: Currently selected session ID

    Returns:
        Tuple of (action, session_id) or None
    """
    is_current = session.id == current_session_id
    title = session.title or f"Chat {session.id[:8]}..."

    # Container for session item
    with st.sidebar.container():
        col1, col2 = st.columns([5, 1])

        with col1:
            # Style current session differently
            button_type = "primary" if is_current else "secondary"
            if st.button(
                title,
                key=f"session_{session.id}",
                use_container_width=True,
                type=button_type,
            ):
                return ("select", session.id)

            # Show metadata
            st.caption(
                f"{session.message_count} msgs | "
                f"{format_datetime(session.created_at)}"
            )

        with col2:
            if st.button("X", key=f"delete_{session.id}", help="Delete session"):
                return ("delete", session.id)

    return None


def render_session_header(session: ChatSessionResponse):
    """
    Render current session header in main area.

    Args:
        session: ChatSessionResponse
    """
    title = session.title or "New Chat"
    st.header(title)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption(f"Messages: {session.message_count}")

    with col2:
        st.caption(f"Created: {format_datetime(session.created_at)}")

    with col3:
        if session.filter_doc_ids:
            st.caption(f"Filtered: {len(session.filter_doc_ids)} docs")
        else:
            st.caption("All documents")


def handle_session_action(
    action: str,
    session_id: Optional[str],
    api_client: APIClient,
) -> Optional[str]:
    """
    Handle session management actions.

    Args:
        action: Action type ("create", "select", "delete")
        session_id: Session ID for select/delete actions
        api_client: APIClient instance

    Returns:
        New current session ID, or None if action failed
    """
    try:
        if action == "create":
            new_session = api_client.create_session()
            st.session_state.messages = []
            return new_session.id

        elif action == "select":
            messages = api_client.get_messages(session_id)
            st.session_state.messages = messages
            return session_id

        elif action == "delete":
            api_client.delete_session(session_id)
            # If deleting current session, clear it
            if st.session_state.get("current_session_id") == session_id:
                st.session_state.messages = []
                return None
            return st.session_state.get("current_session_id")

    except APIError as e:
        st.error(f"Action failed: {e.message}")
        return st.session_state.get("current_session_id")

    return None
