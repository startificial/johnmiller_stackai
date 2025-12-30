"""
Low-Level Mistral Response Generator

Handles direct communication with Mistral AI API for structured response generation.
Implements retry logic with exponential backoff and JSON response parsing.
"""

import asyncio
import json
import re
from typing import List, Optional, Type

from mistralai import Mistral
from pydantic import BaseModel

from backend.app.settings import settings
from backend.app.services.llm_response.exceptions import (
    LLMResponseConfigError,
    LLMResponseGenerationError,
    LLMResponseParseError,
)


class MistralResponseGenerator:
    """
    Low-level Mistral AI response generator.

    Handles direct communication with Mistral API for generating
    structured responses with retry logic and exponential backoff.
    """

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
        Initialize the Mistral response generator.

        Args:
            api_key: Mistral API key (defaults to env var)
            model: Model name for generation (defaults to settings)
            temperature: Generation temperature (defaults to settings)
            max_tokens: Maximum tokens to generate (defaults to settings)
            max_retries: Maximum retry attempts (defaults to settings)
            timeout: API timeout in seconds (defaults to settings)
        """
        llm_settings = settings.llm_response

        self.api_key = api_key or llm_settings.api_key
        self.model = model or llm_settings.model_name
        self.temperature = (
            temperature if temperature is not None else llm_settings.temperature
        )
        self.max_tokens = max_tokens or llm_settings.max_tokens
        self.max_retries = max_retries or llm_settings.max_retries
        self.timeout = timeout or llm_settings.timeout

        if not self.api_key:
            raise LLMResponseConfigError(
                "MISTRAL_API_KEY environment variable is required"
            )

        # Initialize Mistral client
        self._client = Mistral(api_key=self.api_key)

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Type[BaseModel],
    ) -> BaseModel:
        """
        Generate a structured response using Mistral chat completion.

        Args:
            system_prompt: System instructions for the model
            user_prompt: User query and context
            response_schema: Pydantic model for response validation

        Returns:
            Validated Pydantic model instance

        Raises:
            LLMResponseGenerationError: If generation fails after retries
            LLMResponseParseError: If response cannot be parsed
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response_text = await self._chat_completion(messages)
        return self._parse_response(response_text, response_schema)

    async def generate_with_history(
        self,
        system_prompt: str,
        messages: List[dict],
        response_schema: Type[BaseModel],
    ) -> BaseModel:
        """
        Generate a response with conversation history.

        Args:
            system_prompt: System instructions for the model
            messages: List of message dicts with 'role' and 'content'
            response_schema: Pydantic model for response validation

        Returns:
            Validated Pydantic model instance
        """
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        response_text = await self._chat_completion(full_messages)
        return self._parse_response(response_text, response_schema)

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
            LLMResponseGenerationError: If all retries fail
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
                    response_format={"type": "json_object"},  # Force JSON output
                ),
            )

            if response.choices and response.choices[0].message:
                return response.choices[0].message.content

            raise LLMResponseGenerationError("Empty response from Mistral API")

        except LLMResponseGenerationError:
            raise
        except Exception as e:
            if retry_count < self.max_retries:
                # Exponential backoff
                wait_time = 2**retry_count
                await asyncio.sleep(wait_time)
                return await self._chat_completion(messages, retry_count + 1)
            else:
                raise LLMResponseGenerationError(
                    f"Failed to generate response after {self.max_retries} retries: {e}"
                ) from e

    def _parse_response(
        self,
        response_text: str,
        response_schema: Type[BaseModel],
    ) -> BaseModel:
        """
        Parse and validate the model response.

        Args:
            response_text: Raw response from the model
            response_schema: Pydantic model for validation

        Returns:
            Validated Pydantic model instance

        Raises:
            LLMResponseParseError: If parsing or validation fails
        """
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_str = self._extract_json(response_text)
            data = json.loads(json_str)
            return response_schema.model_validate(data)

        except json.JSONDecodeError as e:
            raise LLMResponseParseError(
                f"Failed to parse JSON response: {response_text[:500]}"
            ) from e
        except Exception as e:
            raise LLMResponseParseError(f"Failed to validate response: {e}") from e

    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from response text, handling various formats.

        Args:
            text: Raw text that may contain JSON

        Returns:
            Extracted JSON string
        """
        # Try markdown code block first
        json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if json_match:
            return json_match.group(1)

        # Try to find raw JSON object
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            return json_match.group()

        # Return as-is if no pattern matched
        return text.strip()
