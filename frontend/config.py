"""
Configuration settings for the Streamlit frontend.
"""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Frontend configuration settings."""

    api_base_url: str = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "120"))
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
    page_title: str = "RAG Knowledge Base"
    page_icon: str = "chat"
    layout: str = "wide"


config = Config()
