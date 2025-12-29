"""Helper utility functions."""


def format_response(data: dict) -> dict:
    """Format API response data."""
    return {"data": data, "success": True}

