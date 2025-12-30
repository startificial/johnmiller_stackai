"""
Citation and source display components.
"""

import streamlit as st

from api.models import SourceInfo
from utils.helpers import format_relevance_score, truncate_text


def render_inline_citations(sources: list[SourceInfo]):
    """
    Render inline citation markers with expandable details.

    Args:
        sources: List of SourceInfo objects
    """
    if not sources:
        return

    st.markdown("---")
    st.markdown("**Sources:**")

    for i, source in enumerate(sources, 1):
        title = source.title or f"Source {i}"
        with st.expander(f"[{i}] {truncate_text(title, 50)}"):
            _render_source_detail(source, i)


def _render_source_detail(source: SourceInfo, index: int):
    """
    Render detailed source information.

    Args:
        source: SourceInfo object
        index: Citation number (1-indexed)
    """
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(f"**Document ID:** `{source.document_id[:8]}...`")
        if source.page_number:
            st.markdown(f"**Page:** {source.page_number}")
        st.markdown(f"**Chunk:** `{source.chunk_id}`")

    with col2:
        st.metric(
            label="Relevance",
            value=format_relevance_score(source.relevance_score),
        )

    st.markdown("**Excerpt:**")
    st.text_area(
        label="excerpt",
        value=source.text_excerpt,
        height=150,
        disabled=True,
        label_visibility="collapsed",
        key=f"source_excerpt_{index}_{source.source_id}",
    )


def render_citations_summary(sources: list[SourceInfo]):
    """
    Render a compact citations summary.

    Args:
        sources: List of SourceInfo objects
    """
    if not sources:
        return

    st.caption(f"Based on {len(sources)} source(s)")
    citation_refs = [f"[{i}]" for i in range(1, len(sources) + 1)]
    st.caption(" ".join(citation_refs))


def render_source_chips(sources: list[SourceInfo]):
    """
    Render sources as clickable chips/badges.

    Args:
        sources: List of SourceInfo objects
    """
    if not sources:
        return

    cols = st.columns(min(len(sources), 4))

    for i, (col, source) in enumerate(zip(cols, sources), 1):
        with col:
            relevance = format_relevance_score(source.relevance_score)
            title = source.title or f"Source {i}"
            st.markdown(
                f"**[{i}]** {truncate_text(title, 20)}  \n"
                f"_{relevance}_"
            )
