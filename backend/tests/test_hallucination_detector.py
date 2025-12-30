"""Tests for hallucination detection service.

This module tests:
1. HallucinationDetector service behavior
2. Claim verification logic
3. Hallucination score calculation
4. Integration with LLMResponseService
5. Block and clarify behavior
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from backend.app.services.hallucination_detector import (
    HallucinationDetector,
    MistralHallucinationDetector,
    HallucinationAnalysisResult,
    ClaimVerification,
    HallucinationDetectionError,
    HallucinationConfigError,
    HallucinationAnalysisError,
)
from backend.app.services.llm_response.service import (
    LLMResponseService,
    LLMResponseResult,
)
from backend.app.services.llm_response.schemas import (
    Source,
    HallucinationBlockedResponse,
    ExplainResponse,
    ConfidenceLevel,
)
from backend.app.services.intent_classifier import Intent, IntentResult


class TestClaimVerificationDataclass:
    """Tests for the ClaimVerification dataclass."""

    def test_claim_verification_supported(self):
        """Test creating a supported claim verification."""
        claim = ClaimVerification(
            claim="The system uses AES-256 encryption",
            is_supported=True,
            supporting_source_ids=["src_0", "src_1"],
            reason="",
        )
        assert claim.is_supported is True
        assert len(claim.supporting_source_ids) == 2
        assert claim.to_dict()["claim"] == "The system uses AES-256 encryption"

    def test_claim_verification_unsupported(self):
        """Test creating an unsupported claim verification."""
        claim = ClaimVerification(
            claim="The system supports MFA",
            is_supported=False,
            supporting_source_ids=[],
            reason="No source mentions MFA or multi-factor authentication",
        )
        assert claim.is_supported is False
        assert len(claim.supporting_source_ids) == 0
        assert "MFA" in claim.reason


class TestHallucinationAnalysisResult:
    """Tests for the HallucinationAnalysisResult dataclass."""

    def test_result_passed(self):
        """Test a passing hallucination analysis result."""
        result = HallucinationAnalysisResult(
            response_text="The system uses encryption.",
            claims_checked=1,
            unsupported_claims=[],
            supported_claims=[
                ClaimVerification(
                    claim="uses encryption",
                    is_supported=True,
                    supporting_source_ids=["src_0"],
                    reason="",
                )
            ],
            hallucination_score=0.0,
            passed=True,
        )
        assert result.passed is True
        assert result.hallucination_score == 0.0
        assert len(result.unsupported_claims) == 0

    def test_result_failed(self):
        """Test a failing hallucination analysis result."""
        result = HallucinationAnalysisResult(
            response_text="The system uses encryption and supports MFA.",
            claims_checked=2,
            unsupported_claims=[
                ClaimVerification(
                    claim="supports MFA",
                    is_supported=False,
                    supporting_source_ids=[],
                    reason="No source mentions MFA",
                )
            ],
            supported_claims=[
                ClaimVerification(
                    claim="uses encryption",
                    is_supported=True,
                    supporting_source_ids=["src_0"],
                    reason="",
                )
            ],
            hallucination_score=0.5,  # 1 out of 2 claims unsupported
            passed=False,
        )
        assert result.passed is False
        assert result.hallucination_score == 0.5
        assert len(result.unsupported_claims) == 1

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = HallucinationAnalysisResult(
            response_text="Test response",
            claims_checked=1,
            unsupported_claims=[],
            supported_claims=[],
            hallucination_score=0.0,
            passed=True,
        )
        result_dict = result.to_dict()
        assert "response_text" in result_dict
        assert "hallucination_score" in result_dict
        assert "passed" in result_dict


class TestHallucinationDetectorFormatSources:
    """Tests for the _format_sources helper method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = HallucinationDetector()

    def test_format_empty_sources(self):
        """Test formatting with no sources."""
        result = self.detector._format_sources([])
        assert result == ""

    def test_format_single_source(self):
        """Test formatting a single source."""
        sources = [
            Source(
                source_id="src_0",
                chunk_id="chunk_1",
                document_id="doc_1",
                title="Security Guide",
                text_excerpt="The system uses AES-256 encryption for all data.",
                relevance_score=0.9,
            ),
        ]
        result = self.detector._format_sources(sources)
        assert "[src_0]" in result
        assert "Security Guide" in result
        assert "AES-256" in result

    def test_format_multiple_sources(self):
        """Test formatting multiple sources."""
        sources = [
            Source(
                source_id="src_0",
                chunk_id="chunk_1",
                document_id="doc_1",
                text_excerpt="First source text.",
                relevance_score=0.9,
            ),
            Source(
                source_id="src_1",
                chunk_id="chunk_2",
                document_id="doc_2",
                text_excerpt="Second source text.",
                relevance_score=0.8,
            ),
        ]
        result = self.detector._format_sources(sources)
        assert "[src_0]" in result
        assert "[src_1]" in result
        assert "---" in result  # Separator between sources


class TestHallucinationDetectorCheck:
    """Tests for the hallucination check method."""

    @pytest.mark.asyncio
    async def test_check_returns_passed_when_all_supported(self):
        """Test that check returns passed=True when all claims are supported."""
        detector = HallucinationDetector()

        sources = [
            Source(
                source_id="src_0",
                chunk_id="chunk_1",
                document_id="doc_1",
                text_excerpt="The system uses AES-256 encryption.",
                relevance_score=0.9,
            ),
        ]

        # Mock the low-level detector
        mock_mistral = MagicMock()
        mock_mistral.analyze = AsyncMock(return_value=(
            [],  # No unsupported claims
            [{"claim": "uses AES-256 encryption", "supporting_sources": ["src_0"]}],
            1,
        ))
        detector._detector = mock_mistral

        result = await detector.check(
            response_text="The system uses AES-256 encryption.",
            sources=sources,
        )

        assert result.passed is True
        assert result.hallucination_score == 0.0

    @pytest.mark.asyncio
    async def test_check_returns_failed_when_unsupported_claims(self):
        """Test that check returns passed=False when unsupported claims exist."""
        detector = HallucinationDetector()

        sources = [
            Source(
                source_id="src_0",
                chunk_id="chunk_1",
                document_id="doc_1",
                text_excerpt="The system uses encryption.",
                relevance_score=0.9,
            ),
        ]

        # Mock the low-level detector to return an unsupported claim
        mock_mistral = MagicMock()
        mock_mistral.analyze = AsyncMock(return_value=(
            [{"claim": "supports MFA", "reason": "No source mentions MFA"}],
            [{"claim": "uses encryption", "supporting_sources": ["src_0"]}],
            2,
        ))
        detector._detector = mock_mistral

        result = await detector.check(
            response_text="The system uses encryption and supports MFA.",
            sources=sources,
        )

        assert result.passed is False
        assert result.hallucination_score == 0.5
        assert len(result.unsupported_claims) == 1

    @pytest.mark.asyncio
    async def test_check_with_custom_threshold(self):
        """Test that check respects custom threshold parameter."""
        detector = HallucinationDetector()

        sources = [
            Source(
                source_id="src_0",
                chunk_id="chunk_1",
                document_id="doc_1",
                text_excerpt="Some text.",
                relevance_score=0.9,
            ),
        ]

        mock_mistral = MagicMock()
        mock_mistral.analyze = AsyncMock(return_value=(
            [{"claim": "one unsupported", "reason": "not in sources"}],
            [{"claim": "supported", "supporting_sources": ["src_0"]}],
            2,
        ))
        detector._detector = mock_mistral

        # With default threshold 0.3, score 0.5 should fail
        result1 = await detector.check(
            response_text="Test response.",
            sources=sources,
            threshold=0.3,
        )
        assert result1.passed is False

        # With higher threshold 0.6, score 0.5 should pass
        result2 = await detector.check(
            response_text="Test response.",
            sources=sources,
            threshold=0.6,
        )
        assert result2.passed is True

    @pytest.mark.asyncio
    async def test_check_with_no_sources_returns_passed(self):
        """Test that check with no sources returns passed (nothing to verify against)."""
        detector = HallucinationDetector()

        result = await detector.check(
            response_text="Any response text.",
            sources=[],
        )

        assert result.passed is True
        assert result.claims_checked == 0


class TestHallucinationScoreCalculation:
    """Tests for hallucination score calculation."""

    @pytest.mark.asyncio
    async def test_score_zero_when_all_supported(self):
        """Test score is 0.0 when all claims are supported."""
        detector = HallucinationDetector()

        sources = [Source(
            source_id="src_0", chunk_id="c", document_id="d",
            text_excerpt="text", relevance_score=0.9,
        )]

        mock_mistral = MagicMock()
        mock_mistral.analyze = AsyncMock(return_value=(
            [],  # 0 unsupported
            [{"claim": "a", "supporting_sources": ["src_0"]},
             {"claim": "b", "supporting_sources": ["src_0"]}],  # 2 supported
            2,
        ))
        detector._detector = mock_mistral

        result = await detector.check("test", sources)
        assert result.hallucination_score == 0.0

    @pytest.mark.asyncio
    async def test_score_one_when_all_unsupported(self):
        """Test score is 1.0 when all claims are unsupported."""
        detector = HallucinationDetector()

        sources = [Source(
            source_id="src_0", chunk_id="c", document_id="d",
            text_excerpt="text", relevance_score=0.9,
        )]

        mock_mistral = MagicMock()
        mock_mistral.analyze = AsyncMock(return_value=(
            [{"claim": "a", "reason": "not found"},
             {"claim": "b", "reason": "not found"}],  # 2 unsupported
            [],  # 0 supported
            2,
        ))
        detector._detector = mock_mistral

        result = await detector.check("test", sources)
        assert result.hallucination_score == 1.0

    @pytest.mark.asyncio
    async def test_score_half_when_half_unsupported(self):
        """Test score is 0.5 when half of claims are unsupported."""
        detector = HallucinationDetector()

        sources = [Source(
            source_id="src_0", chunk_id="c", document_id="d",
            text_excerpt="text", relevance_score=0.9,
        )]

        mock_mistral = MagicMock()
        mock_mistral.analyze = AsyncMock(return_value=(
            [{"claim": "unsupported", "reason": "not found"}],  # 1 unsupported
            [{"claim": "supported", "supporting_sources": ["src_0"]}],  # 1 supported
            2,
        ))
        detector._detector = mock_mistral

        result = await detector.check("test", sources)
        assert result.hallucination_score == 0.5

    @pytest.mark.asyncio
    async def test_score_zero_when_no_claims(self):
        """Test score is 0.0 when there are no claims to check."""
        detector = HallucinationDetector()

        sources = [Source(
            source_id="src_0", chunk_id="c", document_id="d",
            text_excerpt="text", relevance_score=0.9,
        )]

        mock_mistral = MagicMock()
        mock_mistral.analyze = AsyncMock(return_value=([], [], 0))
        detector._detector = mock_mistral

        result = await detector.check("test", sources)
        assert result.hallucination_score == 0.0


class TestHallucinationBlockedResponseSchema:
    """Tests for the HallucinationBlockedResponse schema."""

    def test_schema_has_required_fields(self):
        """Test that schema has all expected fields."""
        response = HallucinationBlockedResponse(
            query_understood="User asked about X",
            sources=[],
            clarifying_question="Please rephrase your question",
            unsupported_claims=[],
            hallucination_score=0.5,
            reason="Response blocked due to unverified claims",
        )
        assert response.intent == "hallucination_blocked"
        assert response.needs_clarification is True
        assert response.confidence == ConfidenceLevel.LOW

    def test_schema_with_unsupported_claims(self):
        """Test schema with unsupported claims list."""
        from backend.app.services.llm_response.schemas import UnsupportedClaim

        response = HallucinationBlockedResponse(
            query_understood="User asked about X",
            sources=[],
            clarifying_question="Please rephrase",
            unsupported_claims=[
                UnsupportedClaim(claim="MFA support", reason="Not in sources"),
                UnsupportedClaim(claim="OAuth2", reason="Not mentioned"),
            ],
            hallucination_score=0.67,
            reason="Multiple claims could not be verified",
        )
        assert len(response.unsupported_claims) == 2


class TestLLMResponseServiceHallucinationIntegration:
    """Tests for hallucination detection integration in LLMResponseService."""

    @pytest.mark.asyncio
    async def test_blocks_response_when_hallucination_detected(self):
        """Test that service blocks response when hallucination is detected above threshold."""
        # Create service with mocked generator
        mock_generator = MagicMock()
        mock_generator.generate = AsyncMock(return_value=ExplainResponse(
            query_understood="User asked about encryption",
            confidence=ConfidenceLevel.MEDIUM,
            concept="encryption",
            explanation="The system uses AES-256 and supports MFA.",
        ))
        mock_generator.model = "test-model"

        service = LLMResponseService(generator=mock_generator)

        mock_intent_result = IntentResult(
            query="explain encryption",
            intent=Intent.EXPLAIN,
            confidence=0.9,
            model_used="test-model",
        )

        sources = [
            Source(
                source_id="src_0",
                chunk_id="chunk_1",
                document_id="doc_1",
                text_excerpt="The system uses AES-256 encryption.",
                relevance_score=0.8,
            ),
        ]

        # Mock hallucination detection to fail
        mock_hallucination_result = HallucinationAnalysisResult(
            response_text="The system uses AES-256 and supports MFA.",
            claims_checked=2,
            unsupported_claims=[
                ClaimVerification(
                    claim="supports MFA",
                    is_supported=False,
                    reason="No source mentions MFA",
                )
            ],
            supported_claims=[],
            hallucination_score=0.5,
            passed=False,
        )

        with patch.object(service, '_retrieve_context', new_callable=AsyncMock) as mock_retrieve:
            mock_retrieve.return_value = (sources, ["explain encryption"])

            with patch('backend.app.services.llm_response.service.hallucination_detector') as mock_hd:
                mock_hd.check = AsyncMock(return_value=mock_hallucination_result)

                with patch('backend.app.services.llm_response.service.settings') as mock_settings:
                    mock_settings.hallucination.enabled = True
                    mock_settings.hallucination.threshold = 0.3

                    result = await service.generate_response(
                        query="explain encryption",
                        intent_result=mock_intent_result,
                    )

                    # Verify blocked response was returned
                    assert result.response.intent == "hallucination_blocked"
                    assert isinstance(result.response, HallucinationBlockedResponse)

    @pytest.mark.asyncio
    async def test_proceeds_when_hallucination_check_passes(self):
        """Test that service proceeds normally when hallucination check passes."""
        mock_generator = MagicMock()
        mock_response = ExplainResponse(
            query_understood="User asked about encryption",
            confidence=ConfidenceLevel.HIGH,
            concept="encryption",
            explanation="The system uses AES-256 encryption.",
        )
        mock_generator.generate = AsyncMock(return_value=mock_response)
        mock_generator.model = "test-model"

        service = LLMResponseService(generator=mock_generator)

        mock_intent_result = IntentResult(
            query="explain encryption",
            intent=Intent.EXPLAIN,
            confidence=0.9,
            model_used="test-model",
        )

        sources = [
            Source(
                source_id="src_0",
                chunk_id="chunk_1",
                document_id="doc_1",
                text_excerpt="The system uses AES-256 encryption.",
                relevance_score=0.8,
            ),
        ]

        # Mock hallucination detection to pass
        mock_hallucination_result = HallucinationAnalysisResult(
            response_text="The system uses AES-256 encryption.",
            claims_checked=1,
            unsupported_claims=[],
            supported_claims=[
                ClaimVerification(
                    claim="uses AES-256 encryption",
                    is_supported=True,
                    supporting_source_ids=["src_0"],
                )
            ],
            hallucination_score=0.0,
            passed=True,
        )

        with patch.object(service, '_retrieve_context', new_callable=AsyncMock) as mock_retrieve:
            mock_retrieve.return_value = (sources, ["explain encryption"])

            with patch('backend.app.services.llm_response.service.hallucination_detector') as mock_hd:
                mock_hd.check = AsyncMock(return_value=mock_hallucination_result)

                with patch('backend.app.services.llm_response.service.settings') as mock_settings:
                    mock_settings.hallucination.enabled = True
                    mock_settings.hallucination.threshold = 0.3

                    result = await service.generate_response(
                        query="explain encryption",
                        intent_result=mock_intent_result,
                    )

                    # Verify original response was returned
                    assert result.response.intent == "explain"
                    assert isinstance(result.response, ExplainResponse)

    @pytest.mark.asyncio
    async def test_skips_check_when_disabled(self):
        """Test that hallucination check is skipped when disabled."""
        mock_generator = MagicMock()
        mock_response = ExplainResponse(
            query_understood="Test",
            confidence=ConfidenceLevel.MEDIUM,
            concept="test",
            explanation="Test explanation.",
        )
        mock_generator.generate = AsyncMock(return_value=mock_response)
        mock_generator.model = "test-model"

        service = LLMResponseService(generator=mock_generator)

        mock_intent_result = IntentResult(
            query="test",
            intent=Intent.EXPLAIN,
            confidence=0.9,
            model_used="test-model",
        )

        sources = [
            Source(
                source_id="src_0",
                chunk_id="chunk_1",
                document_id="doc_1",
                text_excerpt="Test source.",
                relevance_score=0.8,
            ),
        ]

        with patch.object(service, '_retrieve_context', new_callable=AsyncMock) as mock_retrieve:
            mock_retrieve.return_value = (sources, ["test"])

            with patch('backend.app.services.llm_response.service.hallucination_detector') as mock_hd:
                with patch('backend.app.services.llm_response.service.settings') as mock_settings:
                    mock_settings.hallucination.enabled = False

                    result = await service.generate_response(
                        query="test",
                        intent_result=mock_intent_result,
                    )

                    # Verify hallucination check was never called
                    mock_hd.check.assert_not_called()
                    # Verify original response was returned
                    assert result.response.intent == "explain"


class TestExtractResponseText:
    """Tests for the _extract_response_text helper method."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch.object(LLMResponseService, '__init__', lambda self, generator=None: None):
            self.service = LLMResponseService()
            self.service._generator = None
            self.service._settings = MagicMock()

    def test_extract_from_explain_response(self):
        """Test extracting text from ExplainResponse."""
        response = ExplainResponse(
            query_understood="Test",
            confidence=ConfidenceLevel.MEDIUM,
            concept="AES encryption",
            explanation="AES is a symmetric encryption algorithm.",
            key_points=["Fast", "Secure", "Widely used"],
        )

        text = self.service._extract_response_text(response)

        assert "AES encryption" in text
        assert "symmetric encryption algorithm" in text
        assert "Fast" in text
        assert "Secure" in text

    def test_extract_from_response_with_steps(self):
        """Test extracting text from response with steps."""
        from backend.app.services.llm_response.schemas import ProcedureResponse, Step

        response = ProcedureResponse(
            query_understood="Test",
            confidence=ConfidenceLevel.MEDIUM,
            task="Deploy application",
            steps=[
                Step(step_number=1, action="Run build command"),
                Step(step_number=2, action="Deploy to server", details="Use SSH"),
            ],
            outcome="Application deployed",
        )

        text = self.service._extract_response_text(response)

        assert "Deploy application" in text
        assert "Run build command" in text
        assert "Deploy to server" in text
        assert "SSH" in text
        assert "Application deployed" in text

    def test_extract_returns_empty_for_minimal_response(self):
        """Test extracting from response with no extractable text fields."""
        # Create a mock response with none of the expected fields
        mock_response = MagicMock()
        mock_response.__dict__ = {}

        # Remove all expected attributes
        for attr in ['explanation', 'summary', 'concept', 'current_status',
                     'problem_summary', 'task', 'comparison_topic',
                     'exploration_area', 'requested_action', 'outcome',
                     'key_points', 'probable_causes', 'recent_changes',
                     'steps', 'solutions']:
            if hasattr(mock_response, attr):
                delattr(mock_response, attr)

        text = self.service._extract_response_text(mock_response)

        assert text == ""


class TestMistralHallucinationDetectorParseResponse:
    """Tests for the response parsing logic in MistralHallucinationDetector."""

    def test_parse_valid_json_response(self):
        """Test parsing a valid JSON response."""
        with patch('backend.app.services.hallucination_detector.settings') as mock_settings:
            mock_settings.hallucination.api_key = "test-key"
            mock_settings.hallucination.model_name = "test-model"
            mock_settings.hallucination.temperature = 0.0
            mock_settings.hallucination.max_tokens = 512
            mock_settings.hallucination.max_retries = 3
            mock_settings.hallucination.timeout = 15.0

            detector = MistralHallucinationDetector(api_key="test-key")

            response_text = '''
            {
                "claims_analyzed": 2,
                "unsupported_claims": [
                    {"claim": "supports MFA", "reason": "Not mentioned"}
                ],
                "supported_claims": [
                    {"claim": "uses encryption", "supporting_sources": ["src_0"]}
                ]
            }
            '''

            unsupported, supported, count = detector._parse_response(response_text)

            assert count == 2
            assert len(unsupported) == 1
            assert len(supported) == 1

    def test_parse_json_with_extra_text(self):
        """Test parsing JSON response with extra text around it."""
        with patch('backend.app.services.hallucination_detector.settings') as mock_settings:
            mock_settings.hallucination.api_key = "test-key"
            mock_settings.hallucination.model_name = "test-model"
            mock_settings.hallucination.temperature = 0.0
            mock_settings.hallucination.max_tokens = 512
            mock_settings.hallucination.max_retries = 3
            mock_settings.hallucination.timeout = 15.0

            detector = MistralHallucinationDetector(api_key="test-key")

            response_text = '''
            Here is my analysis:
            {"claims_analyzed": 1, "unsupported_claims": [], "supported_claims": []}
            That's my response.
            '''

            unsupported, supported, count = detector._parse_response(response_text)

            assert count == 1

    def test_parse_invalid_json_raises_error(self):
        """Test that invalid JSON raises HallucinationAnalysisError."""
        with patch('backend.app.services.hallucination_detector.settings') as mock_settings:
            mock_settings.hallucination.api_key = "test-key"
            mock_settings.hallucination.model_name = "test-model"
            mock_settings.hallucination.temperature = 0.0
            mock_settings.hallucination.max_tokens = 512
            mock_settings.hallucination.max_retries = 3
            mock_settings.hallucination.timeout = 15.0

            detector = MistralHallucinationDetector(api_key="test-key")

            with pytest.raises(HallucinationAnalysisError):
                detector._parse_response("This is not valid JSON at all")


class TestHallucinationConfigError:
    """Tests for configuration error handling."""

    def test_raises_error_when_no_api_key(self):
        """Test that missing API key raises HallucinationConfigError."""
        with patch('backend.app.services.hallucination_detector.settings') as mock_settings:
            mock_settings.hallucination.api_key = ""
            mock_settings.hallucination.model_name = "test-model"
            mock_settings.hallucination.temperature = 0.0
            mock_settings.hallucination.max_tokens = 512
            mock_settings.hallucination.max_retries = 3
            mock_settings.hallucination.timeout = 15.0

            with pytest.raises(HallucinationConfigError):
                MistralHallucinationDetector()
