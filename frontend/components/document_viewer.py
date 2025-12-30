"""
Document viewer component for displaying uploaded files.
"""

from typing import Optional

import streamlit as st

from api.models import IngestionResponse


def render_document_list(documents: list[IngestionResponse]):
    """
    Display list of uploaded documents with metadata.

    Args:
        documents: List of IngestionResponse objects
    """
    if not documents:
        st.info("No documents uploaded yet. Use the upload panel to add documents.")
        return

    st.markdown(f"**{len(documents)} document(s) in session:**")

    for doc in documents:
        with st.expander(f"{doc.filename}", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Pages", doc.page_count)
            col2.metric("Chunks", doc.chunks_created)
            col3.metric("Embedded", doc.chunks_embedded)
            col4.markdown(f"Status: **{doc.status}**")
            st.caption(f"ID: `{doc.document_id}`")


def render_document_filter(
    documents: list[IngestionResponse],
) -> Optional[list[str]]:
    """
    Render multiselect for filtering chat to specific documents.

    Args:
        documents: List of available documents

    Returns:
        List of selected document IDs, or None if none selected
    """
    if not documents:
        return None

    options = {doc.filename: doc.document_id for doc in documents}
    selected = st.multiselect(
        "Filter to specific documents (optional)",
        options=list(options.keys()),
        help="Leave empty to search all documents",
    )

    if selected:
        return [options[name] for name in selected]
    return None


def render_document_summary(documents: list[IngestionResponse]):
    """
    Render a compact summary of documents.

    Args:
        documents: List of IngestionResponse objects
    """
    if not documents:
        st.caption("No documents")
        return

    total_pages = sum(d.page_count for d in documents)
    total_chunks = sum(d.chunks_created for d in documents)

    st.caption(
        f"{len(documents)} docs | {total_pages} pages | {total_chunks} chunks"
    )
