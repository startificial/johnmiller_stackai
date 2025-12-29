"""Tests for utility functions."""

from app.utils.helpers import format_response


def test_format_response():
    """Test response formatting utility."""
    data = {"key": "value"}
    result = format_response(data)
    assert result["success"] is True
    assert result["data"] == data

