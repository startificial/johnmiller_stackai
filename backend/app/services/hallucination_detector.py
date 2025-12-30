"""
Hallucination Detection Service

Post-hoc evidence check that scans LLM response sentences for claims
not supported by the retrieved sources using Mistral AI's fastest
model (ministral-3b-latest) for low-latency inference.

This service analyzes generated responses and verifies each factual claim
against the source documents. If unsupported claims are detected above
the configured threshold, the response is flagged for blocking.

Usage:
    from backend.app.services.hallucination_detector import hallucination_detector

    result = await hallucination_detector.check(
        response_text="The system uses AES-256 encryption.",
        sources=sources_list,
    )
    if not result.passed:
        # Handle hallucination detected
        print(result.unsupported_claims)
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from mistralai import Mistral

from backend.app.settings import settings


class HallucinationDetectionError(Exception):
    """Base exception for hallucination detection errors."""

    pass


class HallucinationConfigError(HallucinationDetectionError):
    """Raised when configuration is invalid or missing."""

    pass


class HallucinationAnalysisError(HallucinationDetectionError):
    """Raised when analysis fails or response cannot be parsed."""

    pass


@dataclass
class ClaimVerification:
    """Verification result for a single claim."""

    claim: str
    is_supported: bool
    supporting_source_ids: List[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "claim": self.claim,
            "is_supported": self.is_supported,
            "supporting_source_ids": self.supporting_source_ids,
            "reason": self.reason,
        }


@dataclass
class HallucinationAnalysisResult:
    """Complete result from hallucination analysis."""

    response_text: str
    claims_checked: int
    unsupported_claims: List[ClaimVerification]
    supported_claims: List[ClaimVerification]
    hallucination_score: float  # 0.0 (no hallucination) to 1.0 (all hallucinated)
    passed: bool  # True if hallucination_score <= threshold

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "response_text": self.response_text,
            "claims_checked": self.claims_checked,
            "unsupported_claims": [c.to_dict() for c in self.unsupported_claims],
            "supported_claims": [c.to_dict() for c in self.supported_claims],
            "hallucination_score": self.hallucination_score,
            "passed": self.passed,
        }


# System prompt for hallucination detection
HALLUCINATION_SYSTEM_PROMPT = """You are a fact-checker for a RAG (Retrieval-Augmented Generation) system. Your task is to verify that claims in an answer are supported by the provided source documents.

Rules for verification:
1. A claim is SUPPORTED if the source explicitly states it, or if it can be directly paraphrased from the source
2. A claim is SUPPORTED if it is a reasonable inference from explicit source information
3. A claim is SUPPORTED when the answer acknowledges conflicting/varying values from different sources - this is honest reporting, not hallucination
4. A claim is SUPPORTED if it accurately summarizes ranges or variations found across sources (e.g., "ranges from X to Y" when those values appear in different sources)
5. A claim is UNSUPPORTED if it adds information not present in any source
6. A claim is UNSUPPORTED if it contradicts information in the sources
7. Ignore filler phrases, greetings, and meta-statements (e.g., "Based on the sources...", "I can help with that")
8. Focus only on factual claims that can be verified against the sources

IMPORTANT: Acknowledging that different sources contain different values is SUPPORTED behavior, not hallucination. The answer is being transparent about source discrepancies.

Respond ONLY with valid JSON in this exact format:
{
  "claims_analyzed": <number>,
  "unsupported_claims": [
    {"claim": "<the unsupported claim>", "reason": "<why it's not supported>"}
  ],
  "supported_claims": [
    {"claim": "<the supported claim>", "supporting_sources": ["src_0", "src_1"]}
  ]
}

If there are no claims to verify (e.g., just a greeting), respond with:
{"claims_analyzed": 0, "unsupported_claims": [], "supported_claims": []}"""


class MistralHallucinationDetector:
    """Low-level Mistral AI hallucination detector using chat completion."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        """
        Initialize the Mistral hallucination detector.

        Args:
            api_key: Mistral API key (defaults to env var)
            model: Model name for detection (defaults to settings)
            temperature: Generation temperature (defaults to settings)
            max_tokens: Maximum tokens to generate (defaults to settings)
            max_retries: Maximum retry attempts (defaults to settings)
            timeout: API timeout in seconds (defaults to settings)
        """
        hd_settings = settings.hallucination

        self.api_key = api_key or hd_settings.api_key
        self.model = model or hd_settings.model_name
        self.temperature = temperature if temperature is not None else hd_settings.temperature
        self.max_tokens = max_tokens or hd_settings.max_tokens
        self.max_retries = max_retries or hd_settings.max_retries
        self.timeout = timeout or hd_settings.timeout

        if not self.api_key:
            raise HallucinationConfigError(
                "MISTRAL_API_KEY environment variable is required"
            )

        # Initialize Mistral client
        self._client = Mistral(api_key=self.api_key)

    async def analyze(
        self,
        response_text: str,
        sources_context: str,
    ) -> tuple[List[dict], List[dict], int]:
        """
        Analyze response text for hallucinations against source context.

        Args:
            response_text: The LLM-generated response text to check
            sources_context: Formatted source documents for verification

        Returns:
            Tuple of (unsupported_claims, supported_claims, claims_analyzed)

        Raises:
            HallucinationAnalysisError: If analysis fails after retries
        """
        user_prompt = f"""## SOURCES
{sources_context}

## ANSWER TO CHECK
{response_text}

Analyze each factual claim in the answer and verify against the sources."""

        messages = [
            {"role": "system", "content": HALLUCINATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        response = await self._chat_completion(messages)
        return self._parse_response(response)

    async def _chat_completion(
        self,
        messages: list[dict],
        retry_count: int = 0,
    ) -> str:
        """
        Execute chat completion with retry logic.

        Args:
            messages: Chat messages for the API
            retry_count: Current retry attempt

        Returns:
            Generated text response

        Raises:
            HallucinationAnalysisError: If all retries fail
        """
        try:
            # Mistral SDK is synchronous, run in thread pool
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.chat.complete(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                ),
            )

            if response.choices and response.choices[0].message:
                return response.choices[0].message.content

            raise HallucinationAnalysisError("Empty response from Mistral API")

        except HallucinationAnalysisError:
            raise
        except Exception as e:
            if retry_count < self.max_retries:
                # Exponential backoff
                wait_time = 2**retry_count
                await asyncio.sleep(wait_time)
                return await self._chat_completion(messages, retry_count + 1)
            else:
                raise HallucinationAnalysisError(
                    f"Failed to analyze for hallucinations after {self.max_retries} retries: {e}"
                ) from e

    def _parse_response(self, response_text: str) -> tuple[List[dict], List[dict], int]:
        """
        Parse the model response into claim verification results.

        Args:
            response_text: Raw response from the model

        Returns:
            Tuple of (unsupported_claims, supported_claims, claims_analyzed)

        Raises:
            HallucinationAnalysisError: If the response cannot be parsed
        """
        try:
            # Handle case where model might include extra text
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
            else:
                data = json.loads(response_text.strip())

            claims_analyzed = int(data.get("claims_analyzed", 0))
            unsupported_claims = data.get("unsupported_claims", [])
            supported_claims = data.get("supported_claims", [])

            # Validate structure
            if not isinstance(unsupported_claims, list):
                unsupported_claims = []
            if not isinstance(supported_claims, list):
                supported_claims = []

            return unsupported_claims, supported_claims, claims_analyzed

        except json.JSONDecodeError as e:
            raise HallucinationAnalysisError(
                f"Failed to parse JSON response: {response_text}"
            ) from e
        except (KeyError, TypeError, ValueError) as e:
            raise HallucinationAnalysisError(
                f"Invalid response format: {response_text}"
            ) from e


class HallucinationDetector:
    """
    High-level hallucination detection service.

    Provides post-hoc verification of LLM responses against retrieved
    sources to detect unsupported claims (hallucinations).
    """

    def __init__(
        self,
        detector: Optional[MistralHallucinationDetector] = None,
    ):
        """
        Initialize the hallucination detector.

        Args:
            detector: MistralHallucinationDetector instance (created lazily if not provided)
        """
        self._detector = detector
        self._settings = settings.hallucination

    @property
    def detector(self) -> MistralHallucinationDetector:
        """Lazily initialize the detector to avoid requiring API key at import."""
        if self._detector is None:
            self._detector = MistralHallucinationDetector()
        return self._detector

    @property
    def enabled(self) -> bool:
        """Check if hallucination detection is enabled."""
        return self._settings.enabled

    @property
    def threshold(self) -> float:
        """Get the hallucination score threshold."""
        return self._settings.threshold

    async def check(
        self,
        response_text: str,
        sources: List,
        threshold: Optional[float] = None,
    ) -> HallucinationAnalysisResult:
        """
        Check a response for hallucinations against provided sources.

        Args:
            response_text: The LLM-generated response text to verify
            sources: List of Source objects from retrieval
            threshold: Optional override for hallucination threshold

        Returns:
            HallucinationAnalysisResult with verification details

        Raises:
            HallucinationDetectionError: If verification fails
        """
        effective_threshold = threshold if threshold is not None else self._settings.threshold

        # Format sources for the verification prompt
        sources_context = self._format_sources(sources)

        # If no sources, all claims are potentially unsupported
        if not sources_context.strip():
            return HallucinationAnalysisResult(
                response_text=response_text,
                claims_checked=0,
                unsupported_claims=[],
                supported_claims=[],
                hallucination_score=0.0,
                passed=True,  # No claims to check
            )

        # Analyze the response
        unsupported_raw, supported_raw, claims_analyzed = await self.detector.analyze(
            response_text=response_text,
            sources_context=sources_context,
        )

        # Convert to ClaimVerification objects
        unsupported_claims = [
            ClaimVerification(
                claim=c.get("claim", ""),
                is_supported=False,
                supporting_source_ids=[],
                reason=c.get("reason", ""),
            )
            for c in unsupported_raw
        ]

        supported_claims = [
            ClaimVerification(
                claim=c.get("claim", ""),
                is_supported=True,
                supporting_source_ids=c.get("supporting_sources", []),
                reason="",
            )
            for c in supported_raw
        ]

        # Calculate hallucination score
        total_claims = len(unsupported_claims) + len(supported_claims)
        if total_claims == 0:
            hallucination_score = 0.0
        else:
            hallucination_score = len(unsupported_claims) / total_claims

        # Determine if check passed
        passed = hallucination_score <= effective_threshold

        return HallucinationAnalysisResult(
            response_text=response_text,
            claims_checked=claims_analyzed,
            unsupported_claims=unsupported_claims,
            supported_claims=supported_claims,
            hallucination_score=hallucination_score,
            passed=passed,
        )

    def _format_sources(self, sources: List) -> str:
        """
        Format source objects into context text for verification.

        Args:
            sources: List of Source objects

        Returns:
            Formatted sources string
        """
        if not sources:
            return ""

        context_parts = []
        for source in sources:
            # Handle both Source objects and dicts
            if hasattr(source, 'source_id'):
                source_id = source.source_id
                text = source.text_excerpt
                title = getattr(source, 'title', None)
            else:
                source_id = source.get('source_id', 'unknown')
                text = source.get('text_excerpt', source.get('text', ''))
                title = source.get('title')

            header = f"[{source_id}]"
            if title:
                header += f" {title}"

            context_parts.append(f"{header}\n{text}")

        return "\n\n---\n\n".join(context_parts)

    async def check_batch(
        self,
        responses: List[tuple[str, List]],
        max_concurrent: int = 3,
    ) -> List[HallucinationAnalysisResult]:
        """
        Check multiple responses for hallucinations concurrently.

        Args:
            responses: List of (response_text, sources) tuples
            max_concurrent: Maximum concurrent API calls

        Returns:
            List of HallucinationAnalysisResult objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def check_with_semaphore(
            response_text: str,
            sources: List,
        ) -> HallucinationAnalysisResult:
            async with semaphore:
                return await self.check(response_text, sources)

        tasks = [check_with_semaphore(text, srcs) for text, srcs in responses]
        return await asyncio.gather(*tasks)


# Singleton instance for easy imports
hallucination_detector = HallucinationDetector()
