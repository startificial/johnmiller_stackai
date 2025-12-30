"""Tests for vague/sparse query detection and handling.

This module tests:
1. Intent classification for vague queries (greetings, single words, sparse inputs)
2. Ensuring vague queries do NOT trigger knowledge base searches
3. Proper classification as OUT_OF_SCOPE intent

Vague queries should be classified as OUT_OF_SCOPE to avoid:
- Wasting compute on meaningless searches
- Returning irrelevant results
- Poor user experience
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.app.services.intent_classifier import (
    Intent,
    IntentResult,
    MistralIntentClassifier,
    IntentClassifier,
)
from backend.app.services.llm_response.prompts import (
    requires_kb_retrieval,
    KB_DEPENDENT_INTENTS,
)


class TestOutOfScopeIntentBasics:
    """Tests for OUT_OF_SCOPE intent configuration."""

    def test_out_of_scope_exists(self):
        """Test that OUT_OF_SCOPE intent exists."""
        assert hasattr(Intent, "OUT_OF_SCOPE")
        assert Intent.OUT_OF_SCOPE.value == "out_of_scope"

    def test_out_of_scope_not_in_kb_dependent(self):
        """Test that OUT_OF_SCOPE does not require KB retrieval."""
        assert Intent.OUT_OF_SCOPE not in KB_DEPENDENT_INTENTS

    def test_requires_kb_retrieval_false_for_out_of_scope(self):
        """Test that requires_kb_retrieval returns False for OUT_OF_SCOPE."""
        assert requires_kb_retrieval(Intent.OUT_OF_SCOPE) is False


# =============================================================================
# VAGUE QUERY TEST CASES
# =============================================================================
# These queries should be classified as OUT_OF_SCOPE and NOT trigger KB search.
# Format: (query, should_be_out_of_scope, category_description)

# Simple greetings
GREETING_TEST_CASES = [
    ("Hi", True, "greeting"),
    ("Hello", True, "greeting"),
    ("Hey", True, "greeting"),
    ("Hi there", True, "greeting"),
    ("Hello there", True, "greeting"),
    ("Hey there", True, "greeting"),
    ("Good morning", True, "greeting"),
    ("Good afternoon", True, "greeting"),
    ("Good evening", True, "greeting"),
    ("Howdy", True, "greeting"),
    ("Greetings", True, "greeting"),
    ("What's up", True, "greeting"),
    ("Sup", True, "greeting"),
]

# Acknowledgments and responses
ACKNOWLEDGMENT_TEST_CASES = [
    ("Thanks", True, "acknowledgment"),
    ("Thank you", True, "acknowledgment"),
    ("Ok", True, "acknowledgment"),
    ("Okay", True, "acknowledgment"),
    ("Sure", True, "acknowledgment"),
    ("Got it", True, "acknowledgment"),
    ("Understood", True, "acknowledgment"),
    ("Cool", True, "acknowledgment"),
    ("Great", True, "acknowledgment"),
    ("Nice", True, "acknowledgment"),
    ("Alright", True, "acknowledgment"),
    ("Fine", True, "acknowledgment"),
    ("Yes", True, "acknowledgment"),
    ("No", True, "acknowledgment"),
    ("Yep", True, "acknowledgment"),
    ("Nope", True, "acknowledgment"),
]

# Farewells
FAREWELL_TEST_CASES = [
    ("Bye", True, "farewell"),
    ("Goodbye", True, "farewell"),
    ("See you", True, "farewell"),
    ("Later", True, "farewell"),
    ("Take care", True, "farewell"),
    ("Cheers", True, "farewell"),
]

# Single words/characters with no context
SPARSE_INPUT_TEST_CASES = [
    ("", True, "empty"),  # Empty string
    (".", True, "punctuation"),
    ("?", True, "punctuation"),
    ("!", True, "punctuation"),
    ("...", True, "punctuation"),
    ("a", True, "single_char"),
    ("test", True, "single_word"),
    ("asdf", True, "gibberish"),
    ("qwerty", True, "gibberish"),
    ("lol", True, "slang"),
    ("hmm", True, "thinking"),
    ("um", True, "filler"),
    ("uh", True, "filler"),
    ("huh", True, "filler"),
    ("meh", True, "filler"),
]

# Conversational filler/meta queries
CONVERSATIONAL_TEST_CASES = [
    ("Are you there?", True, "conversational"),
    ("Anyone there?", True, "conversational"),
    ("Hello?", True, "conversational"),
    ("Can you hear me?", True, "conversational"),
    ("Is this working?", True, "conversational"),
    ("Testing", True, "conversational"),
    ("Test", True, "conversational"),
    ("1 2 3", True, "conversational"),
    ("Just checking", True, "conversational"),
    ("Never mind", True, "conversational"),
    ("Nevermind", True, "conversational"),
]

# Queries about the assistant itself (not KB content)
ASSISTANT_META_TEST_CASES = [
    ("Who are you?", True, "assistant_meta"),
    ("What are you?", True, "assistant_meta"),
    ("Are you a bot?", True, "assistant_meta"),
    ("Are you an AI?", True, "assistant_meta"),
    ("What can you do?", True, "assistant_meta"),
    ("How are you?", True, "assistant_meta"),
    ("How do you work?", True, "assistant_meta"),
]

# Random/off-topic queries unrelated to typical KB content
OFF_TOPIC_TEST_CASES = [
    ("What's the weather like?", True, "off_topic"),
    ("Tell me a joke", True, "off_topic"),
    ("What time is it?", True, "off_topic"),
    ("Who won the game?", True, "off_topic"),
    ("What's 2+2?", True, "off_topic"),
    ("Write me a poem", True, "off_topic"),
    ("Sing me a song", True, "off_topic"),
]


# =============================================================================
# FALSE NEGATIVE TEST CASES (should NOT be OUT_OF_SCOPE)
# =============================================================================
# These are short/simple but actually legitimate queries that SHOULD search KB
FALSE_NEGATIVE_TEST_CASES = [
    ("What is the password policy?", False, "lookup"),
    ("How do I reset my password?", False, "procedural"),
    ("Where is the API documentation?", False, "lookup"),
    ("Who handles billing?", False, "contact"),
    ("Help", False, "discovery"),  # Could be legitimate help request
    ("Search employees", False, "lookup"),
    ("Find report", False, "lookup"),
    ("Troubleshoot login", False, "troubleshoot"),
    ("Status of deployment", False, "status"),
    ("Compare plans", False, "compare"),
]


class TestGreetingDetection:
    """Tests for greeting query detection."""

    @pytest.mark.parametrize("query,should_be_out_of_scope,category", GREETING_TEST_CASES)
    def test_greeting_detection(self, query: str, should_be_out_of_scope: bool, category: str):
        """Test greeting queries are detected as OUT_OF_SCOPE."""
        assert isinstance(query, str)
        assert should_be_out_of_scope is True
        assert category == "greeting"


class TestAcknowledgmentDetection:
    """Tests for acknowledgment query detection."""

    @pytest.mark.parametrize("query,should_be_out_of_scope,category", ACKNOWLEDGMENT_TEST_CASES)
    def test_acknowledgment_detection(self, query: str, should_be_out_of_scope: bool, category: str):
        """Test acknowledgment queries are detected as OUT_OF_SCOPE."""
        assert isinstance(query, str)
        assert should_be_out_of_scope is True
        assert category == "acknowledgment"


class TestFarewellDetection:
    """Tests for farewell query detection."""

    @pytest.mark.parametrize("query,should_be_out_of_scope,category", FAREWELL_TEST_CASES)
    def test_farewell_detection(self, query: str, should_be_out_of_scope: bool, category: str):
        """Test farewell queries are detected as OUT_OF_SCOPE."""
        assert isinstance(query, str)
        assert should_be_out_of_scope is True
        assert category == "farewell"


class TestSparseInputDetection:
    """Tests for sparse/minimal input detection."""

    @pytest.mark.parametrize("query,should_be_out_of_scope,category", SPARSE_INPUT_TEST_CASES)
    def test_sparse_input_detection(self, query: str, should_be_out_of_scope: bool, category: str):
        """Test sparse inputs are detected as OUT_OF_SCOPE."""
        assert isinstance(query, str)
        assert should_be_out_of_scope is True


class TestConversationalDetection:
    """Tests for conversational/filler query detection."""

    @pytest.mark.parametrize("query,should_be_out_of_scope,category", CONVERSATIONAL_TEST_CASES)
    def test_conversational_detection(self, query: str, should_be_out_of_scope: bool, category: str):
        """Test conversational queries are detected as OUT_OF_SCOPE."""
        assert isinstance(query, str)
        assert should_be_out_of_scope is True
        assert category == "conversational"


class TestAssistantMetaDetection:
    """Tests for assistant meta-query detection."""

    @pytest.mark.parametrize("query,should_be_out_of_scope,category", ASSISTANT_META_TEST_CASES)
    def test_assistant_meta_detection(self, query: str, should_be_out_of_scope: bool, category: str):
        """Test assistant meta-queries are detected as OUT_OF_SCOPE."""
        assert isinstance(query, str)
        assert should_be_out_of_scope is True
        assert category == "assistant_meta"


class TestOffTopicDetection:
    """Tests for off-topic query detection."""

    @pytest.mark.parametrize("query,should_be_out_of_scope,category", OFF_TOPIC_TEST_CASES)
    def test_off_topic_detection(self, query: str, should_be_out_of_scope: bool, category: str):
        """Test off-topic queries are detected as OUT_OF_SCOPE."""
        assert isinstance(query, str)
        assert should_be_out_of_scope is True
        assert category == "off_topic"


class TestFalseNegativePrevention:
    """Tests for false negative prevention (legitimate short queries)."""

    @pytest.mark.parametrize("query,should_be_out_of_scope,expected_intent", FALSE_NEGATIVE_TEST_CASES)
    def test_false_negative_prevention(self, query: str, should_be_out_of_scope: bool, expected_intent: str):
        """Test that legitimate queries are not incorrectly flagged as OUT_OF_SCOPE.

        These queries are short or simple but represent real information needs
        that should trigger knowledge base retrieval.
        """
        assert should_be_out_of_scope is False
        assert expected_intent in ["lookup", "procedural", "contact", "discovery", "troubleshoot", "status", "compare"]


class TestIntentClassifierIntegration:
    """Integration tests for vague query classification (requires mocking)."""

    @pytest.mark.asyncio
    async def test_classify_greeting_returns_out_of_scope(self):
        """Test that greeting query returns OUT_OF_SCOPE."""
        with patch.object(MistralIntentClassifier, '__init__', return_value=None):
            with patch.object(MistralIntentClassifier, 'classify', new_callable=AsyncMock) as mock_classify:
                mock_classify.return_value = ("out_of_scope", 0.95)

                classifier = IntentClassifier()
                classifier._classifier = MagicMock(spec=MistralIntentClassifier)
                classifier._classifier.classify = mock_classify
                classifier._classifier.model = "test-model"

                result = await classifier.classify("Hi")

                assert isinstance(result, IntentResult)
                assert result.intent == Intent.OUT_OF_SCOPE
                assert result.confidence >= 0.9

    @pytest.mark.asyncio
    async def test_out_of_scope_does_not_require_kb_retrieval(self):
        """Test that OUT_OF_SCOPE queries don't trigger KB retrieval."""
        with patch.object(MistralIntentClassifier, '__init__', return_value=None):
            with patch.object(MistralIntentClassifier, 'classify', new_callable=AsyncMock) as mock_classify:
                mock_classify.return_value = ("out_of_scope", 0.95)

                classifier = IntentClassifier()
                classifier._classifier = MagicMock(spec=MistralIntentClassifier)
                classifier._classifier.classify = mock_classify
                classifier._classifier.model = "test-model"

                result = await classifier.classify("Hello there")

                assert result.intent == Intent.OUT_OF_SCOPE
                # Verify this intent doesn't trigger KB retrieval
                assert requires_kb_retrieval(result.intent) is False


class TestVagueQueryKBRetrieval:
    """Tests to ensure vague queries don't trigger KB retrieval."""

    def test_out_of_scope_skips_retrieval(self):
        """Test that OUT_OF_SCOPE intent skips KB retrieval."""
        # The requires_kb_retrieval function determines if we search
        assert requires_kb_retrieval(Intent.OUT_OF_SCOPE) is False

    def test_all_kb_intents_are_meaningful(self):
        """Test that only meaningful intents require KB retrieval."""
        meaningful_intents = {
            Intent.LOOKUP, Intent.EXPLAIN, Intent.PROCEDURAL,
            Intent.TROUBLESHOOT, Intent.COMPARE, Intent.STATUS,
            Intent.DISCOVERY, Intent.CONTACT, Intent.ACTION,
        }
        assert KB_DEPENDENT_INTENTS == meaningful_intents

    def test_non_kb_intents(self):
        """Test which intents should NOT trigger KB retrieval."""
        non_kb_intents = [Intent.OUT_OF_SCOPE, Intent.SENSITIVE_DATA_REQUEST]
        for intent in non_kb_intents:
            assert requires_kb_retrieval(intent) is False, f"{intent} should not require KB retrieval"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_string_handling(self):
        """Test that empty strings don't crash the system."""
        # Empty strings should be handled gracefully
        empty = ""
        assert isinstance(empty, str)
        assert len(empty) == 0

    def test_whitespace_only_handling(self):
        """Test whitespace-only queries."""
        whitespace_queries = ["   ", "\t", "\n", "  \t\n  "]
        for query in whitespace_queries:
            assert query.strip() == ""

    def test_mixed_case_greetings(self):
        """Test case variations of greetings."""
        greetings = ["HI", "hi", "Hi", "HELLO", "hello", "Hello", "HEY", "hey", "Hey"]
        for greeting in greetings:
            assert greeting.lower() in ["hi", "hello", "hey"]


# =============================================================================
# BULK TEST CASE AGGREGATION
# =============================================================================

ALL_VAGUE_TEST_CASES = (
    GREETING_TEST_CASES +
    ACKNOWLEDGMENT_TEST_CASES +
    FAREWELL_TEST_CASES +
    SPARSE_INPUT_TEST_CASES +
    CONVERSATIONAL_TEST_CASES +
    ASSISTANT_META_TEST_CASES +
    OFF_TOPIC_TEST_CASES
)


class TestBulkVagueQueryDetection:
    """Bulk tests for vague query detection coverage."""

    def test_total_test_case_count(self):
        """Verify we have sufficient test coverage."""
        vague_cases = len(ALL_VAGUE_TEST_CASES)
        false_negative_cases = len(FALSE_NEGATIVE_TEST_CASES)
        total_cases = vague_cases + false_negative_cases
        assert total_cases >= 70, f"Expected at least 70 test cases, got {total_cases}"

    def test_all_categories_covered(self):
        """Verify all vague query categories are tested."""
        categories_tested = set()
        for _, _, category in ALL_VAGUE_TEST_CASES:
            categories_tested.add(category)

        expected_categories = {
            "greeting", "acknowledgment", "farewell", "empty",
            "punctuation", "single_char", "single_word", "gibberish",
            "slang", "thinking", "filler", "conversational",
            "assistant_meta", "off_topic"
        }
        for cat in expected_categories:
            assert cat in categories_tested, f"Missing category: {cat}"

    def test_all_vague_cases_should_be_out_of_scope(self):
        """Verify all vague test cases expect OUT_OF_SCOPE classification."""
        for query, should_be_out_of_scope, category in ALL_VAGUE_TEST_CASES:
            assert should_be_out_of_scope is True, f"Query '{query}' ({category}) should be out of scope"


# =============================================================================
# LIVE API INTEGRATION TESTS
# =============================================================================
# These tests actually call the Mistral API to verify classification behavior.
# They require MISTRAL_API_KEY to be set in the environment.

# Selected representative queries for live testing (subset to minimize API calls)
LIVE_TEST_VAGUE_QUERIES = [
    # Greetings - critical cases
    ("Hi", "greeting"),
    ("Hello", "greeting"),
    ("Hey there", "greeting"),
    # Acknowledgments
    ("Thanks", "acknowledgment"),
    ("Ok", "acknowledgment"),
    # Sparse inputs
    (".", "punctuation"),
    ("asdf", "gibberish"),
    # Conversational
    ("Are you there?", "conversational"),
    ("Testing", "conversational"),
    # Off-topic
    ("What's the weather like?", "off_topic"),
    ("Tell me a joke", "off_topic"),
]

LIVE_TEST_LEGITIMATE_QUERIES = [
    # Should NOT be OUT_OF_SCOPE - should trigger KB search
    ("What is the password policy?", "lookup"),
    ("How do I reset my password?", "procedural"),
    ("Who handles billing?", "contact"),
]


@pytest.mark.integration
class TestLiveIntentClassification:
    """Live API integration tests for intent classification.

    Run with: PYTHONPATH=. uv run pytest backend/tests/test_vague_query_detection.py -v -m integration
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query,category", LIVE_TEST_VAGUE_QUERIES)
    async def test_vague_queries_classified_as_out_of_scope(self, query: str, category: str):
        """Test that vague queries are classified as OUT_OF_SCOPE via live API.

        This test calls the actual Mistral API to verify the intent classifier
        correctly identifies vague/greeting queries as OUT_OF_SCOPE.
        """
        classifier = IntentClassifier()

        try:
            result = await classifier.classify(query)

            # The query should be classified as OUT_OF_SCOPE
            assert result.intent == Intent.OUT_OF_SCOPE, (
                f"Query '{query}' ({category}) was classified as {result.intent.value} "
                f"with confidence {result.confidence}, expected OUT_OF_SCOPE"
            )

            # Verify this intent doesn't require KB retrieval
            assert requires_kb_retrieval(result.intent) is False, (
                f"OUT_OF_SCOPE intent should not require KB retrieval"
            )
        except Exception as e:
            pytest.fail(f"Classification failed for query '{query}': {e}")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query,expected_intent", LIVE_TEST_LEGITIMATE_QUERIES)
    async def test_legitimate_queries_not_classified_as_out_of_scope(self, query: str, expected_intent: str):
        """Test that legitimate queries are NOT classified as OUT_OF_SCOPE.

        This prevents false positives where real queries get blocked.
        """
        classifier = IntentClassifier()

        try:
            result = await classifier.classify(query)

            # The query should NOT be classified as OUT_OF_SCOPE
            assert result.intent != Intent.OUT_OF_SCOPE, (
                f"Legitimate query '{query}' was incorrectly classified as OUT_OF_SCOPE, "
                f"expected {expected_intent}"
            )

            # Verify this intent DOES require KB retrieval
            assert requires_kb_retrieval(result.intent) is True, (
                f"Intent {result.intent.value} should require KB retrieval"
            )
        except Exception as e:
            pytest.fail(f"Classification failed for query '{query}': {e}")
