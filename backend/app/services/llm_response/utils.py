"""
LLM Response Utilities

Citation rendering, validation, and other helper functions for LLM responses.
"""

import re
from typing import List, Tuple

from backend.app.services.llm_response.schemas import (
    BaseResponse,
    Citation,
    Source,
)
from backend.app.services.llm_response.exceptions import CitationValidationError


def render_response_with_citations(
    response_text: str,
    sources: List[Source],
    link_format: str = "markdown",
) -> str:
    """
    Render response text with inline citation markers expanded.

    Converts [src_0], [src_1] style citations to a readable format.

    Args:
        response_text: Response text with citation markers
        sources: List of source objects
        link_format: Format for rendering - "markdown" or "plain"

    Returns:
        Rendered text with expanded citations
    """
    source_map = {s.source_id: s for s in sources}

    def replace_citation(match):
        source_id = match.group(1)
        if source_id in source_map:
            source = source_map[source_id]
            if link_format == "markdown":
                title = source.title or f"Source {source_id}"
                page_info = f" (p.{source.page_number})" if source.page_number else ""
                return f"[{title}{page_info}]"
            else:
                return f"[{source_id}]"
        return match.group(0)  # Keep original if not found

    # Match patterns like [src_0], [src_1], etc.
    return re.sub(r"\[(src_\d+)\]", replace_citation, response_text)


def validate_citations(
    response: BaseResponse,
) -> Tuple[bool, List[str]]:
    """
    Validate that all citations reference valid sources.

    Args:
        response: Response with citations to validate

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    source_ids = {s.source_id for s in response.sources}

    for i, citation in enumerate(response.citations):
        for source_id in citation.source_ids:
            if source_id not in source_ids:
                errors.append(
                    f"Citation {i}: references unknown source '{source_id}'"
                )

        # Validate citation confidence value
        valid_confidence = {"direct", "paraphrased", "inferred"}
        if citation.confidence not in valid_confidence:
            errors.append(
                f"Citation {i}: invalid confidence '{citation.confidence}' "
                f"(must be one of {valid_confidence})"
            )

    return len(errors) == 0, errors


def extract_citation_indices(text: str) -> List[str]:
    """
    Extract all citation source IDs from text.

    Args:
        text: Text containing [src_N] style citations

    Returns:
        List of unique source IDs found (e.g., ["src_0", "src_2"])
    """
    matches = re.findall(r"\[(src_\d+)\]", text)
    return sorted(set(matches), key=lambda x: int(x.split("_")[1]))


def format_sources_for_display(
    sources: List[Source],
    include_excerpts: bool = False,
    max_excerpt_length: int = 200,
) -> str:
    """
    Format sources as a readable list for display.

    Args:
        sources: List of Source objects
        include_excerpts: Whether to include text excerpts
        max_excerpt_length: Maximum length of excerpts

    Returns:
        Formatted string listing all sources
    """
    if not sources:
        return "No sources."

    lines = ["**Sources:**"]
    for source in sources:
        line = f"- [{source.source_id}]"
        if source.title:
            line += f" {source.title}"
        if source.page_number:
            line += f" (page {source.page_number})"
        line += f" - relevance: {source.relevance_score:.0%}"

        if include_excerpts and source.text_excerpt:
            excerpt = source.text_excerpt[:max_excerpt_length]
            if len(source.text_excerpt) > max_excerpt_length:
                excerpt += "..."
            line += f"\n  > {excerpt}"

        lines.append(line)

    return "\n".join(lines)


def calculate_response_confidence_score(
    sources: List[Source],
    citations: List[Citation],
) -> float:
    """
    Calculate an overall confidence score based on sources and citations.

    Higher scores indicate more reliable responses:
    - More sources = higher confidence
    - Higher relevance scores = higher confidence
    - More "direct" citations = higher confidence

    Args:
        sources: List of sources used
        citations: List of citations made

    Returns:
        Confidence score between 0.0 and 1.0
    """
    if not sources:
        return 0.0

    # Factor 1: Average relevance of sources (0-1)
    avg_relevance = sum(s.relevance_score for s in sources) / len(sources)

    # Factor 2: Number of sources (diminishing returns)
    # 1 source = 0.5, 3+ sources = 1.0
    source_count_factor = min(1.0, 0.3 + (len(sources) * 0.23))

    # Factor 3: Citation quality
    if citations:
        direct_count = sum(1 for c in citations if c.confidence == "direct")
        citation_quality = direct_count / len(citations)
    else:
        citation_quality = 0.5  # Neutral if no citations

    # Weighted combination
    confidence = (
        avg_relevance * 0.4 + source_count_factor * 0.3 + citation_quality * 0.3
    )

    return min(1.0, max(0.0, confidence))


def merge_duplicate_sources(sources: List[Source]) -> List[Source]:
    """
    Merge duplicate sources by chunk_id, keeping highest relevance.

    Args:
        sources: List of potentially duplicate sources

    Returns:
        Deduplicated list of sources
    """
    seen: dict = {}
    for source in sources:
        if source.chunk_id not in seen:
            seen[source.chunk_id] = source
        elif source.relevance_score > seen[source.chunk_id].relevance_score:
            seen[source.chunk_id] = source

    # Re-assign source_ids after deduplication
    result = []
    for i, source in enumerate(sorted(seen.values(), key=lambda s: s.relevance_score, reverse=True)):
        source.source_id = f"src_{i}"
        result.append(source)

    return result
