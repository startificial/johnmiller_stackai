"""
Chat interface components.
"""

from typing import Optional

import streamlit as st

from api.models import ChatMessageResponse, SourceInfo
from components.citations import render_inline_citations
from components.response_renderer import render_response


def render_chat_input() -> Optional[str]:
    """
    Render the chat input box.

    Returns:
        User message if submitted, None otherwise
    """
    return st.chat_input(
        placeholder="Ask a question about your documents...",
        key="chat_input",
    )


def render_user_message(content: str):
    """
    Render a user message.

    Args:
        content: Message content
    """
    with st.chat_message("user"):
        st.markdown(content)


def render_assistant_message(
    content: str,
    intent: Optional[str] = None,
    sources: Optional[list[SourceInfo]] = None,
):
    """
    Render an assistant message with optional citations.

    Args:
        content: Message content (may be JSON string)
        intent: Intent classification
        sources: Citation sources
    """
    with st.chat_message("assistant"):
        # Render the structured response based on intent
        render_response(content, intent)

        # Show sources if available
        if sources:
            render_inline_citations(sources)


def render_message(message: ChatMessageResponse):
    """
    Render a single chat message.

    Args:
        message: ChatMessageResponse object
    """
    if message.role == "user":
        render_user_message(message.content)
    else:
        render_assistant_message(
            content=message.content,
            intent=message.intent,
            sources=message.sources,
        )


def render_chat_history(messages: list[ChatMessageResponse]):
    """
    Render all messages in the conversation.

    Args:
        messages: List of ChatMessageResponse objects
    """
    for message in messages:
        render_message(message)
