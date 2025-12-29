"""
Intent Classification Service

Classifies user queries into intent categories using Mistral AI's fastest
model (ministral-3b-latest) for low-latency inference.

Intent Categories:
- lookup: Finding a specific document, page, or resource they know exists
- explain: Understanding what something is or how it works conceptually
- procedural: Getting step-by-step instructions to accomplish a task
- troubleshoot: Diagnosing or resolving a problem or error
- compare: Evaluating differences between options
- status: Checking current state, progress, or recent updates
- discovery: Exploring what's available or possible
- contact: Finding who to talk to or who owns something
- action: Requesting something be done by a person
- out_of_scope: Unrelated to the knowledge base

Usage:
    from backend.app.services.intent_classifier import intent_classifier

    result = await intent_classifier.classify("How do I reset my password?")
    print(result.intent)       # "procedural"
    print(result.confidence)   # 0.95
"""

import asyncio
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from mistralai import Mistral

from backend.app.settings import settings


class IntentClassificationError(Exception):
    """Base exception for intent classification errors."""

    pass


class IntentConfigError(IntentClassificationError):
    """Raised when configuration is invalid or missing."""

    pass


class IntentParseError(IntentClassificationError):
    """Raised when the model response cannot be parsed."""

    pass


class Intent(str, Enum):
    """Enumeration of valid intent categories."""

    LOOKUP = "lookup"
    EXPLAIN = "explain"
    PROCEDURAL = "procedural"
    TROUBLESHOOT = "troubleshoot"
    COMPARE = "compare"
    STATUS = "status"
    DISCOVERY = "discovery"
    CONTACT = "contact"
    ACTION = "action"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass
class IntentResult:
    """Result of intent classification."""

    query: str
    intent: Intent
    confidence: float
    model_used: str

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "query": self.query,
            "intent": self.intent.value,
            "confidence": self.confidence,
            "model_used": self.model_used,
        }


# System prompt for intent classification
INTENT_SYSTEM_PROMPT = """You are an intent classifier for a RAG (Retrieval-Augmented Generation) system. Your task is to classify user queries into exactly one intent category.

Intent categories:
- lookup: Finding a specific document, page, or resource they know exists
- explain: Understanding what something is or how it works conceptually
- procedural: Getting step-by-step instructions to accomplish a task
- troubleshoot: Diagnosing or resolving a problem or error
- compare: Evaluating differences between options
- status: Checking current state, progress, or recent updates
- discovery: Exploring what's available or possible
- contact: Finding who to talk to or who owns something
- action: Requesting something be done by a person
- out_of_scope: Unrelated to the knowledge base

Respond ONLY with valid JSON in this exact format:
{"intent": "<category>", "confidence": <0.0-1.0>}

The confidence score should reflect how certain you are about the classification:
- 0.9-1.0: Very confident, clear intent
- 0.7-0.9: Confident, likely correct
- 0.5-0.7: Moderate confidence, could be another category
- Below 0.5: Low confidence, ambiguous query"""


class MistralIntentClassifier:
    """Low-level Mistral AI intent classifier using chat completion."""

    # Valid intents for validation
    VALID_INTENTS = {intent.value for intent in Intent}

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
        Initialize the Mistral intent classifier.

        Args:
            api_key: Mistral API key (defaults to env var)
            model: Model name for classification (defaults to settings)
            temperature: Generation temperature (defaults to settings)
            max_tokens: Maximum tokens to generate (defaults to settings)
            max_retries: Maximum retry attempts (defaults to settings)
            timeout: API timeout in seconds (defaults to settings)
        """
        ic_settings = settings.intent_classification

        self.api_key = api_key or ic_settings.api_key
        self.model = model or ic_settings.model_name
        self.temperature = temperature if temperature is not None else ic_settings.temperature
        self.max_tokens = max_tokens or ic_settings.max_tokens
        self.max_retries = max_retries or ic_settings.max_retries
        self.timeout = timeout or ic_settings.timeout

        if not self.api_key:
            raise IntentConfigError(
                "MISTRAL_API_KEY environment variable is required"
            )

        # Initialize Mistral client
        self._client = Mistral(api_key=self.api_key)

    async def classify(self, query: str) -> tuple[str, float]:
        """
        Classify the intent of a query using Mistral chat completion.

        Args:
            query: User query to classify

        Returns:
            Tuple of (intent_name, confidence_score)

        Raises:
            IntentClassificationError: If classification fails after retries
        """
        user_prompt = f"Query: {query}"

        messages = [
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        response_text = await self._chat_completion(messages)
        intent, confidence = self._parse_response(response_text)

        return intent, confidence

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
            IntentClassificationError: If all retries fail
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

            raise IntentClassificationError("Empty response from Mistral API")

        except IntentClassificationError:
            raise
        except Exception as e:
            if retry_count < self.max_retries:
                # Exponential backoff
                wait_time = 2**retry_count
                await asyncio.sleep(wait_time)
                return await self._chat_completion(messages, retry_count + 1)
            else:
                raise IntentClassificationError(
                    f"Failed to classify intent after {self.max_retries} retries: {e}"
                ) from e

    def _parse_response(self, response_text: str) -> tuple[str, float]:
        """
        Parse the model response into intent and confidence.

        Args:
            response_text: Raw response from the model

        Returns:
            Tuple of (intent_name, confidence_score)

        Raises:
            IntentParseError: If the response cannot be parsed
        """
        # Try to extract JSON from response
        try:
            # Handle case where model might include extra text
            json_match = re.search(r'\{[^}]+\}', response_text)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
            else:
                data = json.loads(response_text.strip())

            intent = data.get("intent", "").lower().strip()
            confidence = float(data.get("confidence", 0.0))

            # Validate intent
            if intent not in self.VALID_INTENTS:
                # Try fuzzy matching for common variations
                intent = self._normalize_intent(intent)
                if intent not in self.VALID_INTENTS:
                    raise IntentParseError(f"Invalid intent: {intent}")

            # Clamp confidence to valid range
            confidence = max(0.0, min(1.0, confidence))

            return intent, confidence

        except json.JSONDecodeError as e:
            raise IntentParseError(
                f"Failed to parse JSON response: {response_text}"
            ) from e
        except (KeyError, TypeError, ValueError) as e:
            raise IntentParseError(
                f"Invalid response format: {response_text}"
            ) from e

    def _normalize_intent(self, intent: str) -> str:
        """
        Normalize intent string to handle common variations.

        Args:
            intent: Raw intent string

        Returns:
            Normalized intent string
        """
        # Common variations mapping
        variations = {
            "look_up": "lookup",
            "look-up": "lookup",
            "find": "lookup",
            "search": "lookup",
            "explanation": "explain",
            "understanding": "explain",
            "what_is": "explain",
            "procedure": "procedural",
            "how_to": "procedural",
            "steps": "procedural",
            "instructions": "procedural",
            "troubleshooting": "troubleshoot",
            "debug": "troubleshoot",
            "fix": "troubleshoot",
            "error": "troubleshoot",
            "problem": "troubleshoot",
            "comparison": "compare",
            "comparing": "compare",
            "difference": "compare",
            "vs": "compare",
            "state": "status",
            "progress": "status",
            "current": "status",
            "discover": "discovery",
            "explore": "discovery",
            "available": "discovery",
            "who": "contact",
            "owner": "contact",
            "responsible": "contact",
            "request": "action",
            "do": "action",
            "perform": "action",
            "out-of-scope": "out_of_scope",
            "outofscope": "out_of_scope",
            "unrelated": "out_of_scope",
            "off_topic": "out_of_scope",
        }

        return variations.get(intent, intent)


class IntentClassifier:
    """
    High-level intent classification service.

    Provides intent classification for user queries to enable
    intent-aware routing and response generation.
    """

    def __init__(
        self,
        classifier: Optional[MistralIntentClassifier] = None,
    ):
        """
        Initialize the intent classifier.

        Args:
            classifier: MistralIntentClassifier instance (created lazily if not provided)
        """
        self._classifier = classifier
        self._settings = settings.intent_classification

    @property
    def classifier(self) -> MistralIntentClassifier:
        """Lazily initialize the classifier to avoid requiring API key at import."""
        if self._classifier is None:
            self._classifier = MistralIntentClassifier()
        return self._classifier

    async def classify(self, query: str) -> IntentResult:
        """
        Classify the intent of a user query.

        Args:
            query: User query to classify

        Returns:
            IntentResult containing intent and confidence

        Raises:
            IntentClassificationError: If classification fails
        """
        intent_str, confidence = await self.classifier.classify(query)

        return IntentResult(
            query=query,
            intent=Intent(intent_str),
            confidence=confidence,
            model_used=self.classifier.model,
        )

    async def classify_batch(
        self,
        queries: list[str],
        max_concurrent: int = 5,
    ) -> list[IntentResult]:
        """
        Classify multiple queries concurrently.

        Args:
            queries: List of queries to classify
            max_concurrent: Maximum concurrent API calls

        Returns:
            List of IntentResult objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def classify_with_semaphore(query: str) -> IntentResult:
            async with semaphore:
                return await self.classify(query)

        tasks = [classify_with_semaphore(q) for q in queries]
        return await asyncio.gather(*tasks)


# Singleton instance for easy imports
intent_classifier = IntentClassifier()
