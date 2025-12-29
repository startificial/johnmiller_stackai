"""
Query Transformation Service

Transforms user queries using Mistral's faster models to improve retrieval
performance while minimizing latency.

Strategies implemented:
1. Multi-Query Generation (RAG-Fusion)
   - Generates multiple query variants from different perspectives
   - Each variant may retrieve different relevant documents
   - Results are fused using RRF in the retriever

Usage Flow:
1. User submits original query
2. QueryTransformer generates N query variants
3. Each variant is passed to HybridRetriever
4. Retriever results are fused using RRF
5. Final ranked results returned to user
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import List, Optional

from mistralai import Mistral

from backend.app.settings import settings


class QueryTransformError(Exception):
    """Base exception for query transformation errors."""

    pass


class QueryTransformConfigError(QueryTransformError):
    """Raised when configuration is invalid or missing."""

    pass


class QueryGenerationError(QueryTransformError):
    """Raised when query generation fails."""

    pass


@dataclass
class QueryVariant:
    """A single transformed query variant."""

    text: str
    perspective: str  # Description of the perspective/approach
    index: int  # Position in the variant list (0-indexed)


@dataclass
class MultiQueryResult:
    """Result of multi-query generation."""

    original_query: str
    variants: List[QueryVariant]
    model_used: str
    total_variants: int = field(init=False)

    def __post_init__(self):
        self.total_variants = len(self.variants)

    @property
    def all_queries(self) -> List[str]:
        """Return all query texts including the original."""
        return [self.original_query] + [v.text for v in self.variants]

    @property
    def variant_queries(self) -> List[str]:
        """Return only the variant query texts (excluding original)."""
        return [v.text for v in self.variants]

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "original_query": self.original_query,
            "variants": [
                {
                    "text": v.text,
                    "perspective": v.perspective,
                    "index": v.index,
                }
                for v in self.variants
            ],
            "model_used": self.model_used,
            "total_variants": self.total_variants,
        }


# System prompt for multi-query generation
MULTI_QUERY_SYSTEM_PROMPT = """You are a query transformation expert for a RAG (Retrieval-Augmented Generation) system. Your task is to generate alternative versions of a user's search query to improve document retrieval.

For each query, generate {num_queries} different versions that:
1. Approach the topic from different angles or perspectives
2. Use different terminology (synonyms, related terms)
3. Vary in specificity (some broader, some more specific)
4. Rephrase the question in different ways

Guidelines:
- Keep queries concise and search-friendly
- Maintain the original intent and meaning
- Each variant should be distinct and add value
- Focus on terms likely to appear in relevant documents

Output format (exactly {num_queries} lines, one query per line):
PERSPECTIVE: <brief description> | QUERY: <transformed query>

Example for "What are the health benefits of green tea?":
PERSPECTIVE: Synonym variation | QUERY: health advantages of drinking green tea
PERSPECTIVE: Specific focus | QUERY: antioxidant properties of green tea
PERSPECTIVE: Broader scope | QUERY: green tea nutrition and wellness effects
PERSPECTIVE: Action-oriented | QUERY: how does green tea improve health"""


class MistralQueryGenerator:
    """Low-level Mistral AI query generator using chat completion."""

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
        Initialize the Mistral query generator.

        Args:
            api_key: Mistral API key (defaults to env var)
            model: Model name for generation (defaults to settings)
            temperature: Generation temperature (defaults to settings)
            max_tokens: Maximum tokens to generate (defaults to settings)
            max_retries: Maximum retry attempts (defaults to settings)
            timeout: API timeout in seconds (defaults to settings)
        """
        qt_settings = settings.query_transform

        self.api_key = api_key or qt_settings.api_key
        self.model = model or qt_settings.model_name
        self.temperature = temperature if temperature is not None else qt_settings.temperature
        self.max_tokens = max_tokens or qt_settings.max_tokens
        self.max_retries = max_retries or qt_settings.max_retries
        self.timeout = timeout or qt_settings.timeout

        if not self.api_key:
            raise QueryTransformConfigError(
                "MISTRAL_API_KEY environment variable is required"
            )

        # Initialize Mistral client
        self._client = Mistral(api_key=self.api_key)

    async def generate_query_variants(
        self,
        query: str,
        num_queries: Optional[int] = None,
    ) -> List[QueryVariant]:
        """
        Generate multiple query variants using Mistral chat completion.

        Args:
            query: Original user query
            num_queries: Number of variants to generate (defaults to settings)

        Returns:
            List of QueryVariant objects

        Raises:
            QueryGenerationError: If generation fails after retries
        """
        if num_queries is None:
            num_queries = settings.query_transform.num_queries

        system_prompt = MULTI_QUERY_SYSTEM_PROMPT.format(num_queries=num_queries)
        user_prompt = f"Generate {num_queries} query variants for:\n\n{query}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response_text = await self._chat_completion(messages)
        variants = self._parse_variants(response_text, num_queries)

        return variants

    async def _chat_completion(
        self,
        messages: List[dict],
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
            QueryGenerationError: If all retries fail
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

            raise QueryGenerationError("Empty response from Mistral API")

        except QueryGenerationError:
            raise
        except Exception as e:
            if retry_count < self.max_retries:
                # Exponential backoff
                wait_time = 2**retry_count
                await asyncio.sleep(wait_time)
                return await self._chat_completion(messages, retry_count + 1)
            else:
                raise QueryGenerationError(
                    f"Failed to generate queries after {self.max_retries} retries: {e}"
                ) from e

    def _parse_variants(
        self,
        response_text: str,
        expected_count: int,
    ) -> List[QueryVariant]:
        """
        Parse the model response into QueryVariant objects.

        Args:
            response_text: Raw response from the model
            expected_count: Expected number of variants

        Returns:
            List of parsed QueryVariant objects
        """
        variants = []

        # Pattern to match: PERSPECTIVE: <text> | QUERY: <text>
        pattern = r"PERSPECTIVE:\s*(.+?)\s*\|\s*QUERY:\s*(.+)"

        lines = response_text.strip().split("\n")

        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                perspective = match.group(1).strip()
                query_text = match.group(2).strip()

                variants.append(
                    QueryVariant(
                        text=query_text,
                        perspective=perspective,
                        index=len(variants),
                    )
                )

                if len(variants) >= expected_count:
                    break

        # If parsing failed, try simpler fallback (just numbered lines)
        if len(variants) < expected_count:
            variants = self._fallback_parse(response_text, expected_count)

        return variants[:expected_count]

    def _fallback_parse(
        self,
        response_text: str,
        expected_count: int,
    ) -> List[QueryVariant]:
        """
        Fallback parsing for simpler response formats.

        Handles numbered lists, bullet points, or plain lines.
        """
        variants = []

        # Remove common prefixes like "1.", "-", "*", etc.
        lines = response_text.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove numbering and bullets
            cleaned = re.sub(r"^[\d]+[\.\)]\s*", "", line)
            cleaned = re.sub(r"^[-*]\s*", "", cleaned)
            cleaned = cleaned.strip()

            # Skip if too short or looks like a header
            if len(cleaned) < 10 or cleaned.endswith(":"):
                continue

            variants.append(
                QueryVariant(
                    text=cleaned,
                    perspective=f"Variant {len(variants) + 1}",
                    index=len(variants),
                )
            )

            if len(variants) >= expected_count:
                break

        return variants


class QueryTransformer:
    """
    High-level query transformation service.

    Provides multi-query generation for RAG-Fusion and other
    query transformation strategies to improve retrieval quality.
    """

    def __init__(
        self,
        generator: Optional[MistralQueryGenerator] = None,
    ):
        """
        Initialize the query transformer.

        Args:
            generator: MistralQueryGenerator instance (created lazily if not provided)
        """
        self._generator = generator
        self._settings = settings.query_transform

    @property
    def generator(self) -> MistralQueryGenerator:
        """Lazily initialize the generator to avoid requiring API key at import."""
        if self._generator is None:
            self._generator = MistralQueryGenerator()
        return self._generator

    async def generate_multi_query(
        self,
        query: str,
        num_queries: Optional[int] = None,
        include_original: bool = True,
    ) -> MultiQueryResult:
        """
        Generate multiple query variants for RAG-Fusion.

        This implements the multi-query generation strategy where the original
        query is expanded into multiple semantically similar but lexically
        different queries. Each variant may retrieve different relevant
        documents, and results are then fused using RRF.

        Args:
            query: Original user query
            num_queries: Number of variants to generate (defaults to settings)
            include_original: Whether to include original query in results

        Returns:
            MultiQueryResult containing all variants

        Raises:
            QueryTransformError: If transformation fails
        """
        if num_queries is None:
            num_queries = self._settings.num_queries

        # Generate variants using Mistral
        variants = await self.generator.generate_query_variants(
            query=query,
            num_queries=num_queries,
        )

        return MultiQueryResult(
            original_query=query,
            variants=variants,
            model_used=self.generator.model,
        )

    async def expand_query(
        self,
        query: str,
    ) -> List[str]:
        """
        Convenience method that returns all query strings for retrieval.

        This is a simplified interface for integration with the retriever.
        Returns the original query plus all generated variants.

        Args:
            query: Original user query

        Returns:
            List of query strings (original + variants)
        """
        result = await self.generate_multi_query(query)
        return result.all_queries


# Singleton instance for easy imports
query_transformer = QueryTransformer()
