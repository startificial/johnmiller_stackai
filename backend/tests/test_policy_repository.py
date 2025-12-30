"""Tests for the policy repository service."""

import pytest
from pathlib import Path
import tempfile
import yaml

from backend.app.services.policy_repository import (
    Policy,
    PolicyRepository,
    PolicyLoadError,
    policy_repository,
)


class TestPolicy:
    """Tests for the Policy dataclass."""

    def test_from_dict_complete(self):
        """Test creating Policy from complete dictionary."""
        data = {
            "category": "pii",
            "name": "Test Policy",
            "description": "A test policy",
            "policy_text": "This is the policy text.",
            "keywords": ["keyword1", "keyword2"],
            "examples": ["Example 1", "Example 2"],
        }
        policy = Policy.from_dict(data)
        assert policy.category == "pii"
        assert policy.name == "Test Policy"
        assert policy.description == "A test policy"
        assert policy.policy_text == "This is the policy text."
        assert policy.keywords == ["keyword1", "keyword2"]
        assert policy.examples == ["Example 1", "Example 2"]

    def test_from_dict_minimal(self):
        """Test creating Policy from minimal dictionary."""
        data = {}
        policy = Policy.from_dict(data)
        assert policy.category == ""
        assert policy.name == ""
        assert policy.keywords == []
        assert policy.examples == []


class TestPolicyRepository:
    """Tests for the PolicyRepository class."""

    def test_default_policies_loaded(self):
        """Test that default policies are loaded from YAML files."""
        # The singleton should have loaded the default policies
        all_policies = policy_repository.get_all_policies()
        assert "pii" in all_policies
        assert "legal" in all_policies
        assert "medical" in all_policies
        assert "pci" in all_policies

    def test_get_policy_exists(self):
        """Test getting an existing policy."""
        policy = policy_repository.get_policy("pii")
        assert policy is not None
        assert policy.category == "pii"
        assert "Personally Identifiable Information" in policy.name

    def test_get_policy_not_exists(self):
        """Test getting a non-existent policy."""
        policy = policy_repository.get_policy("nonexistent")
        assert policy is None

    def test_get_policy_case_insensitive(self):
        """Test that policy lookup is case-insensitive."""
        policy = policy_repository.get_policy("PII")
        assert policy is not None
        policy = policy_repository.get_policy("Pii")
        assert policy is not None

    def test_detect_categories_pii(self):
        """Test detecting PII category from query."""
        categories = policy_repository.detect_categories(
            "What is John's social security number?"
        )
        assert "pii" in categories

    def test_detect_categories_legal(self):
        """Test detecting legal category from query."""
        categories = policy_repository.detect_categories(
            "Can I sue for damages?"
        )
        assert "legal" in categories

    def test_detect_categories_medical(self):
        """Test detecting medical category from query."""
        categories = policy_repository.detect_categories(
            "What medication should I take?"
        )
        assert "medical" in categories

    def test_detect_categories_pci(self):
        """Test detecting PCI category from query."""
        categories = policy_repository.detect_categories(
            "Show me the credit card numbers on file"
        )
        assert "pci" in categories

    def test_detect_categories_multiple(self):
        """Test detecting multiple categories from query."""
        # Query that might match both PII and PCI
        categories = policy_repository.detect_categories(
            "Give me customer addresses and credit card numbers"
        )
        assert "pii" in categories
        assert "pci" in categories

    def test_detect_categories_none(self):
        """Test that non-sensitive queries return no categories."""
        categories = policy_repository.detect_categories(
            "What is the weather today?"
        )
        assert len(categories) == 0

    def test_format_policies_for_prompt_single(self):
        """Test formatting a single policy for prompt."""
        formatted = policy_repository.format_policies_for_prompt(["pii"])
        assert "Personally Identifiable Information" in formatted
        assert "Social security numbers" in formatted

    def test_format_policies_for_prompt_multiple(self):
        """Test formatting multiple policies for prompt."""
        formatted = policy_repository.format_policies_for_prompt(["pii", "legal"])
        assert "Personally Identifiable Information" in formatted
        assert "Legal Advice" in formatted

    def test_format_policies_for_prompt_all(self):
        """Test formatting all policies when no categories specified."""
        formatted = policy_repository.format_policies_for_prompt(None)
        # Should include all four policies
        assert "Personally Identifiable" in formatted

    def test_get_policy_summary(self):
        """Test getting policy summary."""
        summary = policy_repository.get_policy_summary("pii")
        assert "Personally Identifiable Information" in summary

    def test_get_policy_summary_not_exists(self):
        """Test getting summary for non-existent policy."""
        summary = policy_repository.get_policy_summary("nonexistent")
        assert summary == ""

    def test_get_all_policy_summaries(self):
        """Test getting all policy summaries."""
        summaries = policy_repository.get_all_policy_summaries(["pii", "legal", "medical", "pci"])
        assert len(summaries) == 4


class TestPolicyRepositoryWithCustomDir:
    """Tests for PolicyRepository with custom directory."""

    def test_load_from_custom_directory(self):
        """Test loading policies from a custom directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test policy file
            policy_data = {
                "category": "pii",
                "name": "Custom PII Policy",
                "description": "A custom policy",
                "policy_text": "Custom policy text.",
                "keywords": ["custom_keyword"],
                "examples": ["Custom example"],
            }
            policy_file = Path(tmpdir) / "pii.yaml"
            with open(policy_file, "w") as f:
                yaml.dump(policy_data, f)

            repo = PolicyRepository(policies_dir=Path(tmpdir))
            policy = repo.get_policy("pii")
            assert policy is not None
            assert policy.name == "Custom PII Policy"

    def test_missing_directory_raises_error(self):
        """Test that missing directory raises PolicyLoadError."""
        with pytest.raises(PolicyLoadError):
            PolicyRepository(policies_dir=Path("/nonexistent/path"))


class TestPolicyKeywordDetection:
    """Detailed tests for keyword detection accuracy."""

    @pytest.mark.parametrize(
        "query,expected_category",
        [
            # PII keywords
            ("Show me the SSN for John", "pii"),
            ("What is their date of birth?", "pii"),
            ("List all employee records", "pii"),
            ("Give me their address", "pii"),
            ("What's their phone number?", "pii"),
            ("Share the email address", "pii"),
            # Legal keywords
            ("Is this contract enforceable?", "legal"),
            ("What are my legal rights?", "legal"),
            ("Can they be held liable?", "legal"),
            ("Should I consult an attorney?", "legal"),
            ("What legal action can I take?", "legal"),
            # Medical keywords
            ("What is the diagnosis?", "medical"),
            ("What treatment do you recommend?", "medical"),
            ("Should I take this medication?", "medical"),
            ("Show me patient records", "medical"),
            ("What are the symptoms?", "medical"),
            # PCI keywords
            ("What is the credit card number?", "pci"),
            ("Show me the CVV", "pci"),
            ("List payment details", "pci"),
            ("What's the card expiration date?", "pci"),
            ("Show me account numbers", "pci"),
        ],
    )
    def test_keyword_detection(self, query: str, expected_category: str):
        """Test that specific keywords are detected correctly."""
        categories = policy_repository.detect_categories(query)
        assert expected_category in categories, f"Expected {expected_category} in {categories} for query: {query}"
