"""
LLM Response Service

Intent-aware response generation for the RAG pipeline using Mistral AI.
Generates structured responses with citations based on intent classification
and retrieved knowledge base context.

Usage:
    from backend.app.services.llm_response import llm_response_service, ConversationTurn

    # Single query
    result = await llm_response_service.generate_response("How do I reset my password?")
    print(result.intent)
    print(result.response.steps)

    # Multi-turn conversation
    history = [
        ConversationTurn(role="user", content="How do I reset my password?"),
        ConversationTurn(role="assistant", content="To reset your password..."),
    ]
    result = await llm_response_service.generate_response(
        query="What if I don't have access to my email?",
        conversation_history=history,
    )
"""

# Service classes
from backend.app.services.llm_response.service import (
    LLMResponseService,
    llm_response_service,
    LLMResponseResult,
)
from backend.app.services.llm_response.generator import MistralResponseGenerator

# Schemas
from backend.app.services.llm_response.schemas import (
    Source,
    Citation,
    ConfidenceLevel,
    ConversationTurn,
    BaseResponse,
    LookupResponse,
    ExplainResponse,
    Step,
    ProcedureResponse,
    Solution,
    TroubleshootResponse,
    ComparisonItem,
    CompareResponse,
    StatusResponse,
    DiscoveryItem,
    DiscoveryResponse,
    Contact,
    ContactResponse,
    ActionResponse,
    OutOfScopeResponse,
    ClarificationResponse,
    IntentResponse,
)

# Prompts
from backend.app.services.llm_response.prompts import (
    SYSTEM_PROMPT,
    CITATION_INSTRUCTIONS,
    INTENT_PROMPTS,
    KB_DEPENDENT_INTENTS,
    get_prompt_for_intent,
    requires_kb_retrieval,
)

# Exceptions
from backend.app.services.llm_response.exceptions import (
    LLMResponseError,
    LLMResponseConfigError,
    LLMResponseGenerationError,
    LLMResponseParseError,
    InsufficientContextError,
    CitationValidationError,
)

# Utilities
from backend.app.services.llm_response.utils import (
    render_response_with_citations,
    validate_citations,
    extract_citation_indices,
    format_sources_for_display,
    calculate_response_confidence_score,
    merge_duplicate_sources,
)

__all__ = [
    # Service
    "LLMResponseService",
    "llm_response_service",
    "LLMResponseResult",
    "MistralResponseGenerator",
    # Schemas
    "Source",
    "Citation",
    "ConfidenceLevel",
    "ConversationTurn",
    "BaseResponse",
    "LookupResponse",
    "ExplainResponse",
    "Step",
    "ProcedureResponse",
    "Solution",
    "TroubleshootResponse",
    "ComparisonItem",
    "CompareResponse",
    "StatusResponse",
    "DiscoveryItem",
    "DiscoveryResponse",
    "Contact",
    "ContactResponse",
    "ActionResponse",
    "OutOfScopeResponse",
    "ClarificationResponse",
    "IntentResponse",
    # Prompts
    "SYSTEM_PROMPT",
    "CITATION_INSTRUCTIONS",
    "INTENT_PROMPTS",
    "KB_DEPENDENT_INTENTS",
    "get_prompt_for_intent",
    "requires_kb_retrieval",
    # Exceptions
    "LLMResponseError",
    "LLMResponseConfigError",
    "LLMResponseGenerationError",
    "LLMResponseParseError",
    "InsufficientContextError",
    "CitationValidationError",
    # Utilities
    "render_response_with_citations",
    "validate_citations",
    "extract_citation_indices",
    "format_sources_for_display",
    "calculate_response_confidence_score",
    "merge_duplicate_sources",
]
