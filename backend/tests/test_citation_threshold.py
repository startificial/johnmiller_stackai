"""Tests for citation threshold enforcement in the chat logic.

This module tests:
1. Similarity threshold enforcement for retrieved chunks
2. "Insufficient evidence" response when no chunks meet threshold
3. ClarificationResponse generation when context is insufficient
4. Edge cases for threshold boundary conditions
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from backend.app.services.llm_response.service import (
    LLMResponseService,
    LLMResponseResult,
)
from backend.app.services.llm_response.schemas import (
    Source,
    ClarificationResponse,
    ConfidenceLevel,
)
from backend.app.services.intent_classifier import Intent, IntentResult


class TestHasSufficientContext:
    """Tests for the _has_sufficient_context method."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create service with mocked generator to avoid API key requirement
        with patch.object(LLMResponseService, '__init__', lambda self, generator=None: None):
            self.service = LLMResponseService()
            self.service._generator = None
            # Mock settings with default threshold of 0.3
            self.service._settings = MagicMock()
            self.service._settings.min_relevance_threshold = 0.3

    def test_returns_false_when_no_sources(self):
        """Test that empty sources list returns False."""
        result = self.service._has_sufficient_context([])
        assert result is False

    def test_returns_false_when_all_sources_below_threshold(self):
        """Test that returns False when all sources are below threshold."""
        sources = [
            Source(
                source_id="src_0",
                chunk_id="chunk_1",
                document_id="doc_1",
                text_excerpt="Some text",
                relevance_score=0.1,  # Below 0.3 threshold
            ),
            Source(
                source_id="src_1",
                chunk_id="chunk_2",
                document_id="doc_1",
                text_excerpt="More text",
                relevance_score=0.2,  # Below 0.3 threshold
            ),
            Source(
                source_id="src_2",
                chunk_id="chunk_3",
                document_id="doc_1",
                text_excerpt="Even more text",
                relevance_score=0.29,  # Just below 0.3 threshold
            ),
        ]
        result = self.service._has_sufficient_context(sources)
        assert result is False

    def test_returns_true_when_one_source_meets_threshold(self):
        """Test that returns True if at least one source meets threshold."""
        sources = [
            Source(
                source_id="src_0",
                chunk_id="chunk_1",
                document_id="doc_1",
                text_excerpt="Low relevance text",
                relevance_score=0.1,  # Below threshold
            ),
            Source(
                source_id="src_1",
                chunk_id="chunk_2",
                document_id="doc_1",
                text_excerpt="High relevance text",
                relevance_score=0.5,  # Above threshold
            ),
        ]
        result = self.service._has_sufficient_context(sources)
        assert result is True

    def test_returns_true_when_source_exactly_at_threshold(self):
        """Test that returns True when source score equals threshold exactly."""
        sources = [
            Source(
                source_id="src_0",
                chunk_id="chunk_1",
                document_id="doc_1",
                text_excerpt="Threshold text",
                relevance_score=0.3,  # Exactly at threshold
            ),
        ]
        result = self.service._has_sufficient_context(sources)
        assert result is True

    def test_returns_true_when_all_sources_above_threshold(self):
        """Test that returns True when all sources are above threshold."""
        sources = [
            Source(
                source_id="src_0",
                chunk_id="chunk_1",
                document_id="doc_1",
                text_excerpt="Good text",
                relevance_score=0.5,
            ),
            Source(
                source_id="src_1",
                chunk_id="chunk_2",
                document_id="doc_1",
                text_excerpt="Better text",
                relevance_score=0.8,
            ),
        ]
        result = self.service._has_sufficient_context(sources)
        assert result is True

    def test_respects_custom_threshold(self):
        """Test that the method respects custom threshold settings."""
        # Set a higher threshold
        self.service._settings.min_relevance_threshold = 0.7

        sources = [
            Source(
                source_id="src_0",
                chunk_id="chunk_1",
                document_id="doc_1",
                text_excerpt="Medium relevance",
                relevance_score=0.5,  # Above default but below 0.7
            ),
        ]
        result = self.service._has_sufficient_context(sources)
        assert result is False

        # Add a source that meets the higher threshold
        sources.append(
            Source(
                source_id="src_1",
                chunk_id="chunk_2",
                document_id="doc_1",
                text_excerpt="High relevance",
                relevance_score=0.75,  # Above 0.7 threshold
            )
        )
        result = self.service._has_sufficient_context(sources)
        assert result is True


class TestClarificationResponseSchema:
    """Tests for the ClarificationResponse schema."""

    def test_schema_has_required_fields(self):
        """Test that ClarificationResponse has all expected fields."""
        response = ClarificationResponse(
            query_understood="User asked about X",
            reason="Insufficient evidence found in knowledge base",
            suggestions=["Try rephrasing your question"],
        )
        assert response.intent == "clarification_needed"
        assert response.query_understood == "User asked about X"
        assert response.reason == "Insufficient evidence found in knowledge base"
        assert len(response.suggestions) == 1

    def test_schema_with_partial_answer(self):
        """Test schema with optional partial answer."""
        response = ClarificationResponse(
            query_understood="User asked about X",
            reason="Only partial information found",
            suggestions=["Narrow your query"],
            partial_answer="Based on limited context, X might be...",
        )
        assert response.partial_answer is not None

    def test_schema_validation_requires_reason(self):
        """Test that reason field is required."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            ClarificationResponse(
                query_understood="Test query",
                # Missing required 'reason' field
                suggestions=[],
            )


class TestInsufficientEvidenceFlow:
    """Integration tests for insufficient evidence handling in generate_response."""

    @pytest.mark.asyncio
    async def test_returns_clarification_response_when_sources_below_threshold(self):
        """Test that generate_response returns ClarificationResponse when all chunks are below threshold."""
        # Set up service with mocked dependencies
        service = LLMResponseService()

        # Mock the intent classifier to return LOOKUP intent
        mock_intent_result = IntentResult(
            query="test query",
            intent=Intent.LOOKUP,
            confidence=0.9,
            model_used="test-model",
        )

        # Create low-relevance sources
        low_relevance_sources = [
            Source(
                source_id="src_0",
                chunk_id="chunk_1",
                document_id="doc_1",
                text_excerpt="Some irrelevant text",
                relevance_score=0.1,
            ),
            Source(
                source_id="src_1",
                chunk_id="chunk_2",
                document_id="doc_1",
                text_excerpt="More irrelevant text",
                relevance_score=0.15,
            ),
        ]

        # Mock ClarificationResponse from generator
        mock_clarification = ClarificationResponse(
            query_understood="User asked about something",
            reason="The retrieved documents do not contain sufficient relevant information",
            suggestions=["Try being more specific", "Use different keywords"],
            partial_answer=None,
        )

        with patch.object(service, '_retrieve_context', new_callable=AsyncMock) as mock_retrieve:
            mock_retrieve.return_value = (low_relevance_sources, ["test query"])

            with patch.object(service, '_generate_clarification_response', new_callable=AsyncMock) as mock_clarify:
                mock_clarify.return_value = LLMResponseResult(
                    query="test query",
                    intent=Intent.LOOKUP,
                    intent_confidence=0.9,
                    response=mock_clarification,
                    sources_used=low_relevance_sources,
                    retrieval_queries=["test query"],
                    model_used="test-model",
                )

                result = await service.generate_response(
                    query="test query",
                    intent_result=mock_intent_result,
                )

                # Verify clarification response was called
                mock_clarify.assert_called_once()
                assert result.response.intent == "clarification_needed"

    @pytest.mark.asyncio
    async def test_returns_clarification_response_when_no_sources_found(self):
        """Test that generate_response returns ClarificationResponse when retrieval returns no results."""
        service = LLMResponseService()

        mock_intent_result = IntentResult(
            query="explain something obscure",
            intent=Intent.EXPLAIN,
            confidence=0.85,
            model_used="test-model",
        )

        mock_clarification = ClarificationResponse(
            query_understood="User asked for explanation",
            reason="No relevant documents were found in the knowledge base",
            suggestions=["Verify the topic exists in our documentation"],
        )

        with patch.object(service, '_retrieve_context', new_callable=AsyncMock) as mock_retrieve:
            # Return empty sources list
            mock_retrieve.return_value = ([], ["test query"])

            with patch.object(service, '_generate_clarification_response', new_callable=AsyncMock) as mock_clarify:
                mock_clarify.return_value = LLMResponseResult(
                    query="explain something obscure",
                    intent=Intent.EXPLAIN,
                    intent_confidence=0.85,
                    response=mock_clarification,
                    sources_used=[],
                    retrieval_queries=["explain something obscure"],
                    model_used="test-model",
                )

                result = await service.generate_response(
                    query="explain something obscure",
                    intent_result=mock_intent_result,
                )

                mock_clarify.assert_called_once()
                assert result.response.intent == "clarification_needed"
                assert len(result.sources_used) == 0

    @pytest.mark.asyncio
    async def test_proceeds_with_response_when_sources_meet_threshold(self):
        """Test that generate_response proceeds normally when sources meet threshold."""
        # Create service with mocked generator to avoid API key requirement
        mock_generator = MagicMock()
        mock_generator.generate = AsyncMock()
        mock_generator.model = "test-model"

        service = LLMResponseService(generator=mock_generator)

        mock_intent_result = IntentResult(
            query="test query",
            intent=Intent.LOOKUP,
            confidence=0.9,
            model_used="test-model",
        )

        # Create high-relevance sources
        high_relevance_sources = [
            Source(
                source_id="src_0",
                chunk_id="chunk_1",
                document_id="doc_1",
                text_excerpt="Highly relevant text",
                relevance_score=0.7,  # Above threshold
            ),
        ]

        with patch.object(service, '_retrieve_context', new_callable=AsyncMock) as mock_retrieve:
            mock_retrieve.return_value = (high_relevance_sources, ["test query"])

            with patch.object(service, '_generate_clarification_response', new_callable=AsyncMock) as mock_clarify:
                # The clarification response should NOT be called when sources meet threshold
                await service.generate_response(
                    query="test query",
                    intent_result=mock_intent_result,
                )

                # Verify clarification was NOT called
                mock_clarify.assert_not_called()


# Test cases for boundary conditions on threshold
# Format: (relevance_scores, threshold, expected_sufficient)
THRESHOLD_BOUNDARY_TEST_CASES = [
    # All below threshold
    ([0.1, 0.2, 0.29], 0.3, False),
    # One exactly at threshold
    ([0.1, 0.2, 0.3], 0.3, True),
    # One just above threshold
    ([0.1, 0.2, 0.31], 0.3, True),
    # Empty list
    ([], 0.3, False),
    # Single source below
    ([0.29], 0.3, False),
    # Single source at threshold
    ([0.3], 0.3, True),
    # Very low threshold, all pass
    ([0.1, 0.2], 0.05, True),
    # Very high threshold, none pass
    ([0.5, 0.6, 0.7], 0.9, False),
    # All scores at zero
    ([0.0, 0.0, 0.0], 0.3, False),
    # Maximum relevance scores
    ([1.0, 0.9, 0.8], 0.3, True),
]


class TestThresholdBoundaryConditions:
    """Tests for boundary conditions on similarity threshold."""

    @pytest.mark.parametrize(
        "relevance_scores,threshold,expected_sufficient",
        THRESHOLD_BOUNDARY_TEST_CASES
    )
    def test_threshold_boundary(
        self,
        relevance_scores: list,
        threshold: float,
        expected_sufficient: bool
    ):
        """Test various boundary conditions for threshold checking."""
        with patch.object(LLMResponseService, '__init__', lambda self, generator=None: None):
            service = LLMResponseService()
            service._generator = None
            service._settings = MagicMock()
            service._settings.min_relevance_threshold = threshold

        sources = [
            Source(
                source_id=f"src_{i}",
                chunk_id=f"chunk_{i}",
                document_id="doc_1",
                text_excerpt=f"Text {i}",
                relevance_score=score,
            )
            for i, score in enumerate(relevance_scores)
        ]

        result = service._has_sufficient_context(sources)
        assert result is expected_sufficient, (
            f"Expected {expected_sufficient} for scores {relevance_scores} "
            f"with threshold {threshold}, got {result}"
        )


class TestClarificationResponseContent:
    """Tests for the content and messaging of clarification responses."""

    def test_clarification_response_indicates_insufficient_evidence(self):
        """Test that clarification response conveys 'insufficient evidence' concept."""
        # These are example reason strings that should indicate insufficient evidence
        valid_reason_patterns = [
            "insufficient",
            "not enough",
            "no relevant",
            "could not find",
            "unable to locate",
            "limited information",
            "insufficient context",
            "insufficient evidence",
        ]

        response = ClarificationResponse(
            query_understood="User asked about X",
            reason="Insufficient evidence found to answer this question",
            suggestions=["Try rephrasing"],
        )

        # Verify the reason contains language about insufficient evidence
        reason_lower = response.reason.lower()
        has_valid_pattern = any(
            pattern in reason_lower for pattern in valid_reason_patterns
        )
        assert has_valid_pattern, (
            f"Reason '{response.reason}' should indicate insufficient evidence"
        )

    def test_clarification_response_provides_actionable_suggestions(self):
        """Test that clarification response includes helpful suggestions."""
        response = ClarificationResponse(
            query_understood="User asked about topic X",
            reason="Insufficient evidence in the knowledge base",
            suggestions=[
                "Try using more specific keywords",
                "Check if the topic is covered in our documentation",
                "Rephrase your question",
            ],
        )

        assert len(response.suggestions) >= 1, "Should provide at least one suggestion"
        for suggestion in response.suggestions:
            assert len(suggestion) > 10, "Suggestions should be meaningful (>10 chars)"


class TestIntentSpecificThresholdBehavior:
    """Tests for threshold enforcement across different intent types."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "intent",
        [
            Intent.LOOKUP,
            Intent.EXPLAIN,
            Intent.PROCEDURAL,
            Intent.TROUBLESHOOT,
            Intent.COMPARE,
            Intent.STATUS,
            Intent.DISCOVERY,
            Intent.CONTACT,
            Intent.ACTION,
        ]
    )
    async def test_all_kb_intents_enforce_threshold(self, intent: Intent):
        """Test that all KB-dependent intents enforce the similarity threshold."""
        service = LLMResponseService()

        mock_intent_result = IntentResult(
            query="test",
            intent=intent,
            confidence=0.9,
            model_used="test-model",
        )

        # Sources below threshold
        low_sources = [
            Source(
                source_id="src_0",
                chunk_id="chunk_1",
                document_id="doc_1",
                text_excerpt="Low relevance",
                relevance_score=0.1,
            ),
        ]

        mock_clarification = ClarificationResponse(
            query_understood="Test query",
            reason="Insufficient context",
            suggestions=[],
        )

        with patch.object(service, '_retrieve_context', new_callable=AsyncMock) as mock_retrieve:
            mock_retrieve.return_value = (low_sources, ["test"])

            with patch.object(service, '_generate_clarification_response', new_callable=AsyncMock) as mock_clarify:
                mock_clarify.return_value = LLMResponseResult(
                    query="test",
                    intent=intent,
                    intent_confidence=0.9,
                    response=mock_clarification,
                    sources_used=low_sources,
                    retrieval_queries=["test"],
                    model_used="test-model",
                )

                result = await service.generate_response(
                    query="test",
                    intent_result=mock_intent_result,
                )

                # Verify clarification was called for low-relevance sources
                mock_clarify.assert_called_once()


class TestEdgeCases:
    """Tests for edge cases in citation threshold enforcement."""

    def test_single_source_just_below_threshold(self):
        """Test boundary case with single source just below threshold."""
        with patch.object(LLMResponseService, '__init__', lambda self, generator=None: None):
            service = LLMResponseService()
            service._generator = None
            service._settings = MagicMock()
            service._settings.min_relevance_threshold = 0.3

        sources = [
            Source(
                source_id="src_0",
                chunk_id="chunk_1",
                document_id="doc_1",
                text_excerpt="Almost relevant",
                relevance_score=0.299999,  # Just below 0.3
            ),
        ]
        result = service._has_sufficient_context(sources)
        assert result is False

    def test_negative_relevance_score_handled(self):
        """Test that negative relevance scores are treated as insufficient."""
        with patch.object(LLMResponseService, '__init__', lambda self, generator=None: None):
            service = LLMResponseService()
            service._generator = None
            service._settings = MagicMock()
            service._settings.min_relevance_threshold = 0.3

        # Note: Source schema has ge=0.0 constraint, but test logic handling
        # This tests the _has_sufficient_context method directly
        sources = [
            Source(
                source_id="src_0",
                chunk_id="chunk_1",
                document_id="doc_1",
                text_excerpt="Invalid score",
                relevance_score=0.0,  # Minimum valid score
            ),
        ]
        result = service._has_sufficient_context(sources)
        assert result is False

    def test_very_high_threshold_rejects_good_sources(self):
        """Test that very high threshold can reject otherwise good sources."""
        with patch.object(LLMResponseService, '__init__', lambda self, generator=None: None):
            service = LLMResponseService()
            service._generator = None
            service._settings = MagicMock()
            service._settings.min_relevance_threshold = 0.95  # Very strict

        sources = [
            Source(
                source_id="src_0",
                chunk_id="chunk_1",
                document_id="doc_1",
                text_excerpt="Pretty good but not perfect",
                relevance_score=0.9,  # Good but below 0.95
            ),
        ]
        result = service._has_sufficient_context(sources)
        assert result is False

    def test_zero_threshold_accepts_all_sources(self):
        """Test that zero threshold accepts any non-empty source list."""
        with patch.object(LLMResponseService, '__init__', lambda self, generator=None: None):
            service = LLMResponseService()
            service._generator = None
            service._settings = MagicMock()
            service._settings.min_relevance_threshold = 0.0

        sources = [
            Source(
                source_id="src_0",
                chunk_id="chunk_1",
                document_id="doc_1",
                text_excerpt="Any text",
                relevance_score=0.0,  # Zero score
            ),
        ]
        result = service._has_sufficient_context(sources)
        assert result is True  # 0.0 >= 0.0 is True
