"""
Policy Repository Service

Loads and provides access to data protection policies from YAML files.
Policies define restrictions for PII, legal, medical, and PCI data requests.

Usage:
    from backend.app.services.policy_repository import policy_repository

    # Get a specific policy
    pii_policy = policy_repository.get_policy("pii")

    # Detect relevant categories from a query
    categories = policy_repository.detect_categories("show me SSN numbers")

    # Format policies for inclusion in LLM prompt
    policy_text = policy_repository.format_policies_for_prompt(["pii", "legal"])
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


class PolicyLoadError(Exception):
    """Raised when a policy file cannot be loaded."""

    pass


@dataclass
class Policy:
    """Represents a data protection policy."""

    category: str  # pii, legal, medical, pci
    name: str  # Human-readable name
    description: str  # Short description
    policy_text: str  # Full policy statement
    keywords: list[str] = field(default_factory=list)  # Detection keywords
    examples: list[str] = field(default_factory=list)  # Example restricted items

    @classmethod
    def from_dict(cls, data: dict) -> "Policy":
        """Create a Policy from a dictionary (parsed YAML)."""
        return cls(
            category=data.get("category", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            policy_text=data.get("policy_text", ""),
            keywords=data.get("keywords", []),
            examples=data.get("examples", []),
        )


class PolicyRepository:
    """Loads and provides access to data protection policies from YAML files."""

    # Valid policy categories
    VALID_CATEGORIES = {"pii", "legal", "medical", "pci"}

    def __init__(self, policies_dir: Optional[Path] = None):
        """
        Initialize the policy repository.

        Args:
            policies_dir: Directory containing policy YAML files.
                         Defaults to backend/app/policies/
        """
        if policies_dir is None:
            # Default to the policies directory relative to this file
            policies_dir = Path(__file__).parent.parent / "policies"

        self._policies_dir = policies_dir
        self._policies: dict[str, Policy] = {}
        self._keyword_pattern: Optional[re.Pattern] = None
        self._load_policies()

    def _load_policies(self) -> None:
        """Load all policy YAML files from the policies directory."""
        if not self._policies_dir.exists():
            raise PolicyLoadError(f"Policies directory not found: {self._policies_dir}")

        # Load each policy file
        for category in self.VALID_CATEGORIES:
            policy_file = self._policies_dir / f"{category}.yaml"
            if policy_file.exists():
                try:
                    with open(policy_file, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                    if data:
                        policy = Policy.from_dict(data)
                        self._policies[category] = policy
                except yaml.YAMLError as e:
                    raise PolicyLoadError(
                        f"Failed to parse policy file {policy_file}: {e}"
                    ) from e
                except OSError as e:
                    raise PolicyLoadError(
                        f"Failed to read policy file {policy_file}: {e}"
                    ) from e

        # Build keyword pattern for category detection
        self._build_keyword_pattern()

    def _build_keyword_pattern(self) -> None:
        """Build a regex pattern from all policy keywords for efficient matching."""
        all_keywords = []
        for policy in self._policies.values():
            all_keywords.extend(policy.keywords)

        if all_keywords:
            # Escape special regex chars and create case-insensitive pattern
            escaped_keywords = [re.escape(kw) for kw in all_keywords]
            pattern_str = r"\b(" + "|".join(escaped_keywords) + r")\b"
            self._keyword_pattern = re.compile(pattern_str, re.IGNORECASE)

    def get_policy(self, category: str) -> Optional[Policy]:
        """
        Get a specific policy by category.

        Args:
            category: The policy category (pii, legal, medical, pci)

        Returns:
            Policy object if found, None otherwise
        """
        return self._policies.get(category.lower())

    def get_all_policies(self) -> dict[str, Policy]:
        """
        Get all loaded policies.

        Returns:
            Dictionary mapping category to Policy object
        """
        return self._policies.copy()

    def detect_categories(self, query: str) -> list[str]:
        """
        Detect which policy categories might be relevant to a query.

        Uses keyword matching to identify potential policy violations.

        Args:
            query: The user query to analyze

        Returns:
            List of matching category names (e.g., ["pii", "legal"])
        """
        detected = set()
        query_lower = query.lower()

        for category, policy in self._policies.items():
            for keyword in policy.keywords:
                if keyword.lower() in query_lower:
                    detected.add(category)
                    break  # One match per category is enough

        return sorted(detected)

    def format_policies_for_prompt(
        self, categories: Optional[list[str]] = None
    ) -> str:
        """
        Format policies as text for inclusion in LLM prompt.

        Args:
            categories: Specific categories to include. If None, includes all.

        Returns:
            Formatted policy text string
        """
        if categories is None:
            categories = list(self._policies.keys())

        policy_texts = []
        for category in categories:
            policy = self._policies.get(category.lower())
            if policy:
                policy_texts.append(
                    f"### {policy.name}\n"
                    f"{policy.policy_text.strip()}\n"
                    f"\nExamples of restricted items:\n"
                    + "\n".join(f"- {ex}" for ex in policy.examples)
                )

        return "\n\n".join(policy_texts)

    def get_policy_summary(self, category: str) -> str:
        """
        Get a one-line summary of a policy for response formatting.

        Args:
            category: The policy category

        Returns:
            Summary string or empty string if policy not found
        """
        policy = self._policies.get(category.lower())
        if policy:
            return f"{policy.name}: {policy.description}"
        return ""

    def get_all_policy_summaries(self, categories: list[str]) -> list[str]:
        """
        Get summaries for multiple policies.

        Args:
            categories: List of category names

        Returns:
            List of summary strings
        """
        summaries = []
        for category in categories:
            summary = self.get_policy_summary(category)
            if summary:
                summaries.append(summary)
        return summaries


# Singleton instance for easy imports
policy_repository = PolicyRepository()
