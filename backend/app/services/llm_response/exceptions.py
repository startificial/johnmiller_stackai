"""
LLM Response Service Exceptions

Custom exception hierarchy for the LLM response generation service.
Follows the established pattern from other services in the codebase.
"""


class LLMResponseError(Exception):
    """Base exception for LLM response generation errors."""

    pass


class LLMResponseConfigError(LLMResponseError):
    """Raised when configuration is invalid or missing."""

    pass


class LLMResponseGenerationError(LLMResponseError):
    """Raised when response generation fails after retries."""

    pass


class LLMResponseParseError(LLMResponseError):
    """Raised when structured response parsing fails."""

    pass


class InsufficientContextError(LLMResponseError):
    """Raised when retrieved context is insufficient for the query."""

    pass


class CitationValidationError(LLMResponseError):
    """Raised when citation validation fails."""

    pass
