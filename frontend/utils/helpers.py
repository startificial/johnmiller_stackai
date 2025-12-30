"""
Utility functions and helpers for the frontend.
"""

import asyncio
import json
from typing import Any


def run_async(coro):
    """
    Run an async coroutine in a sync context.

    Streamlit is synchronous, so we need this bridge for async operations.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def parse_response_content(content: str) -> dict[str, Any]:
    """
    Parse JSON response content from assistant messages.

    The backend may store structured responses as JSON strings.

    Args:
        content: Message content (may be JSON string or plain text)

    Returns:
        Parsed dict or {"raw_content": content} if not valid JSON
    """
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {"raw_content": content}


def format_relevance_score(score: float) -> str:
    """
    Format relevance score as a percentage string.

    Args:
        score: Score between 0 and 1

    Returns:
        Formatted percentage string (e.g., "85%")
    """
    return f"{score * 100:.0f}%"


def truncate_text(text: str, max_length: int = 200) -> str:
    """
    Truncate text to max length with ellipsis.

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        Truncated text with "..." if needed
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def format_datetime(dt) -> str:
    """
    Format datetime for display.

    Args:
        dt: datetime object

    Returns:
        Formatted string (e.g., "Dec 30, 2024 2:30 PM")
    """
    return dt.strftime("%b %d, %Y %I:%M %p")
