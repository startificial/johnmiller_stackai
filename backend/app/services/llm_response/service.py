"""
High-Level LLM Response Service

Orchestrates the RAG response pipeline: intent classification, query transformation,
retrieval, and structured response generation with citations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from backend.app.services.intent_classifier import (
    Intent,
    IntentResult,
    intent_classifier,
)
from backend.app.services.query_transformer import query_transformer
from backend.app.services.retriever import retriever, RetrievalResult
from backend.app.services.hallucination_detector import (
    hallucination_detector,
    HallucinationAnalysisResult,
)
from backend.app.settings import settings

from backend.app.services.llm_response.generator import MistralResponseGenerator
from backend.app.services.llm_response.schemas import (
    # API Response schemas (include sources/citations)
    Source,
    ConversationTurn,
    LookupResponse,
    ExplainResponse,
    ProcedureResponse,
    TroubleshootResponse,
    CompareResponse,
    StatusResponse,
    DiscoveryResponse,
    ContactResponse,
    ActionResponse,
    SensitiveDataResponse,
    OutOfScopeResponse,
    ClarificationResponse,
    HallucinationBlockedResponse,
    UnsupportedClaim,
    ConfidenceLevel,
    IntentResponse,
    # LLM Response schemas (no sources/citations - for structured output)
    BaseLLMResponse,
    LookupLLMResponse,
    ExplainLLMResponse,
    ProcedureLLMResponse,
    TroubleshootLLMResponse,
    CompareLLMResponse,
    StatusLLMResponse,
    DiscoveryLLMResponse,
    ContactLLMResponse,
    ActionLLMResponse,
)
from backend.app.services.policy_repository import policy_repository
from backend.app.services.llm_response.prompts import (
    SYSTEM_PROMPT,
    CITATION_INSTRUCTIONS,
    CLARIFICATION_PROMPT,
    get_prompt_for_intent,
    requires_kb_retrieval,
)
from backend.app.services.llm_response.exceptions import (
    LLMResponseError,
    InsufficientContextError,
)


@dataclass
class LLMResponseResult:
    """Complete result from LLM response generation."""

    query: str
    intent: Intent
    intent_confidence: float
    response: IntentResponse
    sources_used: List[Source]
    retrieval_queries: List[str]
    model_used: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "query": self.query,
            "intent": self.intent.value,
            "intent_confidence": self.intent_confidence,
            "response": self.response.model_dump() if hasattr(self.response, 'model_dump') else self.response.__dict__,
            "sources_used": [s.model_dump() for s in self.sources_used],
            "retrieval_queries": self.retrieval_queries,
            "model_used": self.model_used,
        }


# Intent to API Response Schema mapping (includes sources/citations)
INTENT_RESPONSE_SCHEMAS: Dict[Intent, Type[BaseModel]] = {
    Intent.LOOKUP: LookupResponse,
    Intent.EXPLAIN: ExplainResponse,
    Intent.PROCEDURAL: ProcedureResponse,
    Intent.TROUBLESHOOT: TroubleshootResponse,
    Intent.COMPARE: CompareResponse,
    Intent.STATUS: StatusResponse,
    Intent.DISCOVERY: DiscoveryResponse,
    Intent.CONTACT: ContactResponse,
    Intent.ACTION: ActionResponse,
    Intent.SENSITIVE_DATA_REQUEST: SensitiveDataResponse,
    Intent.OUT_OF_SCOPE: OutOfScopeResponse,
}

# Intent to LLM Response Schema mapping (for structured output - no sources/citations)
INTENT_LLM_SCHEMAS: Dict[Intent, Type[BaseLLMResponse]] = {
    Intent.LOOKUP: LookupLLMResponse,
    Intent.EXPLAIN: ExplainLLMResponse,
    Intent.PROCEDURAL: ProcedureLLMResponse,
    Intent.TROUBLESHOOT: TroubleshootLLMResponse,
    Intent.COMPARE: CompareLLMResponse,
    Intent.STATUS: StatusLLMResponse,
    Intent.DISCOVERY: DiscoveryLLMResponse,
    Intent.CONTACT: ContactLLMResponse,
    Intent.ACTION: ActionLLMResponse,
}


class LLMResponseService:
    """
    High-level LLM response service for the RAG pipeline.

    Orchestrates intent classification, query transformation, retrieval,
    and response generation to produce intent-aware responses with citations.
    """

    def __init__(
        self,
        generator: Optional[MistralResponseGenerator] = None,
    ):
        """
        Initialize the LLM response service.

        Args:
            generator: MistralResponseGenerator instance (created lazily if not provided)
        """
        self._generator = generator
        self._settings = settings.llm_response

    @property
    def generator(self) -> MistralResponseGenerator:
        """Lazily initialize the generator to avoid requiring API key at import."""
        if self._generator is None:
            self._generator = MistralResponseGenerator()
        return self._generator

    async def generate_response(
        self,
        query: str,
        intent_result: Optional[IntentResult] = None,
        conversation_history: Optional[List[ConversationTurn]] = None,
        filter_doc_ids: Optional[List[str]] = None,
    ) -> LLMResponseResult:
        """
        Generate an intent-aware response for a user query.

        This is the main entry point for the RAG response pipeline:
        1. Classify intent (if not provided)
        2. Transform query into variants (for KB-dependent intents)
        3. Retrieve relevant context using hybrid search
        4. Generate structured response with citations

        Args:
            query: User query text
            intent_result: Pre-computed intent classification (optional)
            conversation_history: Previous conversation turns for context
            filter_doc_ids: Restrict retrieval to specific documents

        Returns:
            LLMResponseResult with complete response data

        Raises:
            LLMResponseError: If response generation fails
        """
        # Step 1: Get intent classification
        if intent_result is None:
            intent_result = await intent_classifier.classify(query)

        intent = intent_result.intent

        # Get both schema types: LLM schema for generation, API schema for response
        api_response_schema = INTENT_RESPONSE_SCHEMAS[intent]
        llm_schema = INTENT_LLM_SCHEMAS.get(intent)  # None for special intents

        # Step 2: Handle special intents without retrieval
        if intent == Intent.OUT_OF_SCOPE:
            return await self._generate_out_of_scope_response(
                query, intent_result, conversation_history
            )

        if intent == Intent.SENSITIVE_DATA_REQUEST:
            return await self._generate_sensitive_data_response(
                query, intent_result, conversation_history
            )

        # Step 3: Get retrieval context for KB-dependent intents
        sources: List[Source] = []
        retrieval_queries: List[str] = [query]

        if requires_kb_retrieval(intent):
            sources, retrieval_queries = await self._retrieve_context(
                query, filter_doc_ids
            )

            # Check if we have sufficient context
            if not self._has_sufficient_context(sources):
                return await self._generate_clarification_response(
                    query, intent_result, sources, conversation_history
                )

        # Step 4: Build prompt and generate response
        context_text = self._format_context(sources)
        conversation_context = self._format_conversation_history(conversation_history)
        prompt_template = get_prompt_for_intent(intent)

        # Build the user prompt with all context
        user_prompt = prompt_template.format(
            citation_instructions=CITATION_INSTRUCTIONS,
            context=context_text,
            conversation_context=conversation_context,
            query=query,
            current_date=datetime.now().strftime("%Y-%m-%d"),
        )

        # Generate response using LLM schema (no sources/citations)
        if conversation_history:
            messages = self._build_messages_with_history(
                conversation_history, user_prompt
            )
            llm_response = await self.generator.generate_with_history(
                system_prompt=SYSTEM_PROMPT,
                messages=messages,
                response_schema=llm_schema,
            )
        else:
            llm_response = await self.generator.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_schema=llm_schema,
            )

        # Convert LLM response to API response by adding sources/citations
        response = api_response_schema(
            **llm_response.model_dump(),
            sources=sources,
            citations=[],  # Citations not extracted from response text for now
        )

        # Step 5: Post-hoc hallucination check
        hallucination_settings = settings.hallucination
        if hallucination_settings.enabled and sources:
            response_text = self._extract_response_text(response)
            if response_text:
                hallucination_result = await hallucination_detector.check(
                    response_text=response_text,
                    sources=sources,
                )

                if not hallucination_result.passed:
                    return await self._generate_hallucination_blocked_response(
                        query=query,
                        intent_result=intent_result,
                        sources=sources,
                        hallucination_result=hallucination_result,
                        retrieval_queries=retrieval_queries,
                    )

        return LLMResponseResult(
            query=query,
            intent=intent,
            intent_confidence=intent_result.confidence,
            response=response,
            sources_used=sources,
            retrieval_queries=retrieval_queries,
            model_used=self.generator.model,
        )

    async def _retrieve_context(
        self,
        query: str,
        filter_doc_ids: Optional[List[str]] = None,
    ) -> tuple[List[Source], List[str]]:
        """
        Retrieve context using multi-query RAG-Fusion.

        Args:
            query: Original user query
            filter_doc_ids: Optional document filter

        Returns:
            Tuple of (sources list, all query variants used)
        """
        # Generate query variants
        multi_query_result = await query_transformer.generate_multi_query(query)
        all_queries = multi_query_result.all_queries

        # Retrieve for each query variant and deduplicate
        all_results: Dict[str, RetrievalResult] = {}

        for variant_query in all_queries:
            try:
                retrieval_response = await retriever.search(
                    query=variant_query,
                    filter_doc_ids=filter_doc_ids,
                    top_k=self._settings.retrieval_top_k,
                )

                # Deduplicate by chunk_id, keeping highest score
                for result in retrieval_response.results:
                    if result.chunk_id not in all_results:
                        all_results[result.chunk_id] = result
                    elif result.rrf_score > all_results[result.chunk_id].rrf_score:
                        all_results[result.chunk_id] = result

            except Exception:
                # Continue with other queries if one fails
                continue

        # Sort by score and take top N
        sorted_results = sorted(
            all_results.values(), key=lambda r: r.rrf_score, reverse=True
        )[: self._settings.max_context_chunks]

        # Convert to Source objects
        sources = []
        for i, r in enumerate(sorted_results):
            sources.append(
                Source(
                    source_id=f"src_{i}",
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    title=r.metadata.get("title"),
                    text_excerpt=r.parent_text or r.text,  # Prefer parent context
                    relevance_score=r.rrf_score,
                    page_number=r.page_number,
                    doc_type=r.metadata.get("doc_type"),
                    last_updated=r.metadata.get("last_updated"),
                )
            )

        return sources, all_queries

    def _has_sufficient_context(self, sources: List[Source]) -> bool:
        """
        Check if retrieved sources provide sufficient context.

        Args:
            sources: Retrieved source objects

        Returns:
            True if context is sufficient, False otherwise
        """
        if not sources:
            return False

        # Check if any source meets the minimum relevance threshold
        threshold = self._settings.min_relevance_threshold
        return any(s.relevance_score >= threshold for s in sources)

    def _format_context(self, sources: List[Source]) -> str:
        """
        Format sources into context text for the prompt.

        Args:
            sources: List of Source objects

        Returns:
            Formatted context string
        """
        if not sources:
            return "No relevant context found in the knowledge base."

        context_parts = []
        for source in sources:
            header = f"[{source.source_id}]"
            if source.title:
                header += f" {source.title}"
            if source.page_number:
                header += f" (page {source.page_number})"
            header += f" - relevance: {source.relevance_score:.2f}"

            context_parts.append(f"{header}\n{source.text_excerpt}")

        return "\n\n---\n\n".join(context_parts)

    def _format_conversation_history(
        self,
        history: Optional[List[ConversationTurn]],
    ) -> str:
        """
        Format conversation history for the prompt.

        Args:
            history: List of conversation turns

        Returns:
            Formatted conversation context string
        """
        if not history:
            return ""

        parts = ["## Conversation History"]
        for turn in history:
            role = "User" if turn.role == "user" else "Assistant"
            parts.append(f"**{role}**: {turn.content}")

        return "\n".join(parts)

    def _build_messages_with_history(
        self,
        history: List[ConversationTurn],
        current_prompt: str,
    ) -> List[dict]:
        """
        Build message list including conversation history.

        Args:
            history: Previous conversation turns
            current_prompt: Current user prompt with context

        Returns:
            List of message dicts for the API
        """
        messages = []

        # Add history
        for turn in history:
            messages.append({"role": turn.role, "content": turn.content})

        # Add current query with full context
        messages.append({"role": "user", "content": current_prompt})

        return messages

    async def _generate_out_of_scope_response(
        self,
        query: str,
        intent_result: IntentResult,
        conversation_history: Optional[List[ConversationTurn]] = None,
    ) -> LLMResponseResult:
        """Generate response for out-of-scope queries without retrieval."""
        prompt_template = get_prompt_for_intent(Intent.OUT_OF_SCOPE)
        conversation_context = self._format_conversation_history(conversation_history)

        user_prompt = prompt_template.format(
            query=query,
            conversation_context=conversation_context,
        )

        response = await self.generator.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=OutOfScopeResponse,
        )

        return LLMResponseResult(
            query=query,
            intent=Intent.OUT_OF_SCOPE,
            intent_confidence=intent_result.confidence,
            response=response,
            sources_used=[],
            retrieval_queries=[],
            model_used=self.generator.model,
        )

    async def _generate_sensitive_data_response(
        self,
        query: str,
        intent_result: IntentResult,
        conversation_history: Optional[List[ConversationTurn]] = None,
    ) -> LLMResponseResult:
        """Generate response for sensitive data requests without retrieval.

        Uses the policy repository to include relevant policies in the prompt.
        """
        prompt_template = get_prompt_for_intent(Intent.SENSITIVE_DATA_REQUEST)
        conversation_context = self._format_conversation_history(conversation_history)

        # Detect which policy categories might be relevant based on the query
        detected_categories = policy_repository.detect_categories(query)

        # If no categories detected by keywords, include all for safety
        if not detected_categories:
            detected_categories = list(policy_repository.get_all_policies().keys())

        # Format policies for inclusion in the prompt
        policies_text = policy_repository.format_policies_for_prompt(detected_categories)

        user_prompt = prompt_template.format(
            policies=policies_text,
            detected_categories=", ".join(detected_categories),
            query=query,
            conversation_context=conversation_context,
        )

        response = await self.generator.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=SensitiveDataResponse,
        )

        return LLMResponseResult(
            query=query,
            intent=Intent.SENSITIVE_DATA_REQUEST,
            intent_confidence=intent_result.confidence,
            response=response,
            sources_used=[],
            retrieval_queries=[],
            model_used=self.generator.model,
        )

    async def _generate_clarification_response(
        self,
        query: str,
        intent_result: IntentResult,
        sources: List[Source],
        conversation_history: Optional[List[ConversationTurn]] = None,
    ) -> LLMResponseResult:
        """Generate clarification request when context is insufficient."""
        conversation_context = self._format_conversation_history(conversation_history)
        context_text = self._format_context(sources) if sources else "No relevant context found."

        user_prompt = CLARIFICATION_PROMPT.format(
            context=context_text,
            query=query,
            conversation_context=conversation_context,
        )

        response = await self.generator.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=ClarificationResponse,
        )

        return LLMResponseResult(
            query=query,
            intent=intent_result.intent,
            intent_confidence=intent_result.confidence,
            response=response,
            sources_used=sources,
            retrieval_queries=[query],
            model_used=self.generator.model,
        )

    def _extract_response_text(self, response: IntentResponse) -> str:
        """
        Extract the main text content from an intent-specific response.

        Different response types have different text fields. This method
        extracts the relevant text for hallucination verification.

        Args:
            response: The generated response object

        Returns:
            Concatenated text content for verification
        """
        text_parts = []

        # Common fields across response types
        if hasattr(response, 'query_understood'):
            # Skip query_understood as it's just a paraphrase
            pass

        # Intent-specific main text fields
        if hasattr(response, 'explanation'):
            text_parts.append(response.explanation)
        if hasattr(response, 'summary'):
            text_parts.append(response.summary)
        if hasattr(response, 'concept'):
            text_parts.append(response.concept)
        if hasattr(response, 'current_status'):
            text_parts.append(response.current_status)
        if hasattr(response, 'problem_summary'):
            text_parts.append(response.problem_summary)
        if hasattr(response, 'task'):
            text_parts.append(response.task)
        if hasattr(response, 'comparison_topic'):
            text_parts.append(response.comparison_topic)
        if hasattr(response, 'exploration_area'):
            text_parts.append(response.exploration_area)
        if hasattr(response, 'requested_action'):
            text_parts.append(response.requested_action)
        if hasattr(response, 'outcome'):
            text_parts.append(response.outcome)

        # List fields
        if hasattr(response, 'key_points'):
            text_parts.extend(response.key_points or [])
        if hasattr(response, 'probable_causes'):
            text_parts.extend(response.probable_causes or [])
        if hasattr(response, 'recent_changes'):
            text_parts.extend(response.recent_changes or [])

        # Steps (for procedural responses)
        if hasattr(response, 'steps') and response.steps:
            for step in response.steps:
                if hasattr(step, 'action'):
                    text_parts.append(step.action)
                if hasattr(step, 'details') and step.details:
                    text_parts.append(step.details)

        # Solutions (for troubleshoot responses)
        if hasattr(response, 'solutions') and response.solutions:
            for sol in response.solutions:
                if hasattr(sol, 'description'):
                    text_parts.append(sol.description)

        return " ".join(filter(None, text_parts))

    async def _generate_hallucination_blocked_response(
        self,
        query: str,
        intent_result: IntentResult,
        sources: List[Source],
        hallucination_result: HallucinationAnalysisResult,
        retrieval_queries: List[str],
    ) -> LLMResponseResult:
        """
        Generate a blocked response when hallucination is detected.

        Instead of returning the hallucinated response, this returns a
        HallucinationBlockedResponse asking the user to clarify.

        Args:
            query: Original user query
            intent_result: Intent classification result
            sources: Retrieved sources
            hallucination_result: Hallucination analysis result
            retrieval_queries: Query variants used for retrieval

        Returns:
            LLMResponseResult with HallucinationBlockedResponse
        """
        # Convert unsupported claims to schema format
        unsupported = [
            UnsupportedClaim(claim=c.claim, reason=c.reason)
            for c in hallucination_result.unsupported_claims
        ]

        # Build clarifying question
        if unsupported:
            claims_summary = ", ".join(c.claim[:50] for c in unsupported[:3])
            clarifying_question = (
                f"I found some information but couldn't fully verify my response "
                f"against the sources. Could you rephrase your question or ask "
                f"about specific aspects? The following claims couldn't be verified: "
                f"{claims_summary}"
            )
        else:
            clarifying_question = (
                "I couldn't generate a fully verified response based on the available "
                "sources. Could you try rephrasing your question or being more specific?"
            )

        response = HallucinationBlockedResponse(
            query_understood=f"I understood you asked about: {query}",
            confidence=ConfidenceLevel.LOW,
            sources=sources,
            citations=[],
            needs_clarification=True,
            clarifying_question=clarifying_question,
            fallback_suggestion=(
                "Try asking about a specific aspect mentioned in the sources, "
                "or check if the information you need might be in a different document."
            ),
            unsupported_claims=unsupported,
            hallucination_score=hallucination_result.hallucination_score,
            reason=(
                f"Response blocked: {len(unsupported)} claim(s) could not be "
                f"verified against the retrieved sources (score: "
                f"{hallucination_result.hallucination_score:.2f}, "
                f"threshold: {settings.hallucination.threshold:.2f})"
            ),
        )

        return LLMResponseResult(
            query=query,
            intent=intent_result.intent,
            intent_confidence=intent_result.confidence,
            response=response,
            sources_used=sources,
            retrieval_queries=retrieval_queries,
            model_used=self.generator.model,
        )


# Singleton instance for easy imports
llm_response_service = LLMResponseService()
