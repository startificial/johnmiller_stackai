"""
LLM Response Schemas

Pydantic v2 schemas for structured LLM response generation with citation support.
Each intent type has a dedicated response schema inheriting from BaseResponse.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    """Confidence level for response accuracy based on source quality."""

    HIGH = "high"  # Multiple corroborating sources
    MEDIUM = "medium"  # Single strong source or partial coverage
    LOW = "low"  # Inferred or incomplete sources
    UNCERTAIN = "uncertain"  # Insufficient context to answer confidently


class Source(BaseModel):
    """Reference to a retrieved source chunk from the knowledge base."""

    source_id: str = Field(..., description="Unique identifier (e.g., 'src_0')")
    chunk_id: str = Field(..., description="Original chunk ID from retrieval")
    document_id: str = Field(..., description="Parent document identifier")
    title: Optional[str] = Field(None, description="Document or section title")
    text_excerpt: str = Field(..., description="Relevant excerpt from the chunk")
    relevance_score: float = Field(
        ..., ge=0.0, le=1.0, description="Retrieval relevance score"
    )
    page_number: Optional[int] = Field(None, description="Page number if available")
    doc_type: Optional[str] = Field(
        None, description="Document type (doc, page, runbook, etc.)"
    )
    last_updated: Optional[str] = Field(
        None, description="When source was last modified"
    )


class Citation(BaseModel):
    """Links a specific claim to its supporting source(s)."""

    claim: str = Field(..., description="The specific statement being cited")
    source_ids: List[str] = Field(
        ..., description="References to Source.source_id values"
    )
    quote: Optional[str] = Field(None, description="Direct quote if applicable")
    confidence: str = Field(
        "direct",
        description="Citation type: 'direct', 'paraphrased', or 'inferred'",
    )


class BaseResponse(BaseModel):
    """Base response model with common citation infrastructure."""

    query_understood: str = Field(..., description="Paraphrase of user's query")
    confidence: ConfidenceLevel = Field(
        ..., description="Overall confidence in the response"
    )
    sources: List[Source] = Field(
        default_factory=list, description="All sources consulted"
    )
    citations: List[Citation] = Field(
        default_factory=list, description="Specific claim-to-source mappings"
    )
    needs_clarification: bool = Field(
        False, description="Whether clarification is needed"
    )
    clarifying_question: Optional[str] = Field(
        None, description="Question to ask user if clarification needed"
    )
    fallback_suggestion: Optional[str] = Field(
        None, description="Alternative suggestion if unable to fully answer"
    )


class LookupResponse(BaseResponse):
    """Response for lookup intent - finding specific documents or resources."""

    intent: str = Field(default="lookup", description="Intent type")
    found: bool = Field(..., description="Whether the resource was found")
    resource_title: Optional[str] = Field(None, description="Title of found resource")
    resource_url: Optional[str] = Field(None, description="URL or path to resource")
    resource_type: Optional[str] = Field(
        None, description="Type: doc, page, file, tool, api, etc."
    )
    summary: str = Field(..., description="Brief summary of what was found or not")
    related_resources: List[Dict[str, str]] = Field(
        default_factory=list, description="Related resources with title and source_id"
    )


class ExplainResponse(BaseResponse):
    """Response for explain intent - conceptual understanding."""

    intent: str = Field(default="explain", description="Intent type")
    concept: str = Field(..., description="The concept being explained")
    explanation: str = Field(
        ..., description="Clear explanation with inline [source_id] citations"
    )
    key_points: List[str] = Field(
        default_factory=list, description="Key takeaways from the explanation"
    )
    analogy: Optional[str] = Field(
        None, description="Optional analogy for understanding"
    )
    related_concepts: List[str] = Field(
        default_factory=list, description="Related concepts to explore"
    )
    technical_depth: str = Field(
        "intermediate", description="Depth: basic, intermediate, or advanced"
    )


class Step(BaseModel):
    """A single step in a procedure."""

    step_number: int = Field(..., description="Step number (1-indexed)")
    action: str = Field(..., description="What to do")
    details: Optional[str] = Field(None, description="Additional context")
    warning: Optional[str] = Field(None, description="Caution or warning if any")
    code_snippet: Optional[str] = Field(None, description="Code if applicable")


class ProcedureResponse(BaseResponse):
    """Response for procedural intent - step-by-step instructions."""

    intent: str = Field(default="procedural", description="Intent type")
    task: str = Field(..., description="The task being explained")
    prerequisites: List[str] = Field(
        default_factory=list, description="What's needed before starting"
    )
    steps: List[Step] = Field(..., min_length=1, description="Ordered steps")
    estimated_time: Optional[str] = Field(None, description="Time estimate if known")
    outcome: str = Field(..., description="What success looks like")
    common_errors: List[str] = Field(
        default_factory=list, description="Mistakes to avoid"
    )
    next_steps: List[str] = Field(
        default_factory=list, description="Follow-up actions"
    )


class Solution(BaseModel):
    """A potential solution for troubleshooting."""

    description: str = Field(..., description="Solution summary")
    steps: List[str] = Field(..., description="Steps to implement")
    likelihood: str = Field("possible", description="likely, possible, or rare")


class TroubleshootResponse(BaseResponse):
    """Response for troubleshoot intent - problem diagnosis and resolution."""

    intent: str = Field(default="troubleshoot", description="Intent type")
    problem_summary: str = Field(..., description="One-line problem description")
    symptoms_identified: List[str] = Field(
        default_factory=list, description="Observed symptoms"
    )
    diagnostic_questions: List[str] = Field(
        default_factory=list, description="Questions to narrow down the issue"
    )
    probable_causes: List[str] = Field(
        default_factory=list, description="Likely causes"
    )
    solutions: List[Solution] = Field(..., min_length=1, description="Potential fixes")
    escalation_path: Optional[str] = Field(None, description="Who to contact if stuck")
    related_issues: List[str] = Field(
        default_factory=list, description="Similar known issues"
    )


class ComparisonItem(BaseModel):
    """A single item being compared."""

    name: str = Field(..., description="Item name")
    summary: str = Field(..., description="Brief description")
    pros: List[str] = Field(default_factory=list, description="Advantages")
    cons: List[str] = Field(default_factory=list, description="Disadvantages")
    best_for: str = Field(..., description="Ideal use case")


class CompareResponse(BaseResponse):
    """Response for compare intent - evaluating differences between options."""

    intent: str = Field(default="compare", description="Intent type")
    comparison_topic: str = Field(..., description="What is being compared")
    items: List[ComparisonItem] = Field(
        ..., min_length=2, description="Items being compared"
    )
    dimensions: List[str] = Field(
        default_factory=list, description="Comparison criteria"
    )
    comparison_table: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description="Structured comparison: {dimension: {item: value}}",
    )
    recommendation: Optional[str] = Field(
        None, description="Recommendation if warranted"
    )
    decision_factors: List[str] = Field(
        default_factory=list, description="Questions to help decide"
    )


class StatusResponse(BaseResponse):
    """Response for status intent - checking current state or progress."""

    intent: str = Field(default="status", description="Intent type")
    subject: str = Field(..., description="What is being reported on")
    current_status: str = Field(..., description="The current status")
    status_type: str = Field(
        "unknown", description="active, completed, blocked, deprecated, unknown"
    )
    last_updated: Optional[str] = Field(None, description="When source was updated")
    source_freshness: str = Field(
        "unknown", description="current, recent, stale, or unknown"
    )
    recent_changes: List[str] = Field(
        default_factory=list, description="Recent changes"
    )
    next_expected_update: Optional[str] = Field(None, description="Next update if known")
    owner: Optional[str] = Field(None, description="Responsible party")
    staleness_warning: bool = Field(
        False, description="Whether source data may be outdated"
    )


class DiscoveryItem(BaseModel):
    """A single discovered item or option."""

    name: str = Field(..., description="Item name")
    description: str = Field(..., description="What it is/does")
    relevance: str = Field(..., description="Why this might be useful")
    link: Optional[str] = Field(None, description="URL if available")


class DiscoveryResponse(BaseResponse):
    """Response for discovery intent - exploring what's available."""

    intent: str = Field(default="discovery", description="Intent type")
    exploration_area: str = Field(..., description="What is being explored")
    items_found: List[DiscoveryItem] = Field(
        default_factory=list, description="Discovered items"
    )
    categories: List[str] = Field(default_factory=list, description="Groupings found")
    suggested_filters: List[str] = Field(
        default_factory=list, description="Ways to narrow down"
    )
    follow_up_queries: List[str] = Field(
        default_factory=list, description="Suggested next questions"
    )
    coverage_note: str = Field(
        ..., description="How complete this list is"
    )


class Contact(BaseModel):
    """Contact information for a person or team."""

    name: str = Field(..., description="Person or team name")
    role: Optional[str] = Field(None, description="Their role")
    team: Optional[str] = Field(None, description="Team name")
    email: Optional[str] = Field(None, description="Email if available")
    slack_channel: Optional[str] = Field(None, description="Slack channel if available")
    relationship_to_query: str = Field(
        "contact", description="owner, expert, escalation, or delegate"
    )


class ContactResponse(BaseResponse):
    """Response for contact intent - finding people or owners."""

    intent: str = Field(default="contact", description="Intent type")
    subject: str = Field(..., description="What/who is being looked up")
    primary_contact: Optional[Contact] = Field(
        None, description="Main contact if found"
    )
    alternative_contacts: List[Contact] = Field(
        default_factory=list, description="Other contacts"
    )
    escalation_path: Optional[str] = Field(None, description="How to escalate")
    contact_method: str = Field(
        ..., description="Recommended way to reach out"
    )
    office_hours: Optional[str] = Field(None, description="Availability if known")
    data_source: str = Field(
        "documentation",
        description="org_chart, ownership_registry, documentation, or inferred",
    )


class ActionResponse(BaseResponse):
    """Response for action intent - requesting something be done."""

    intent: str = Field(default="action", description="Intent type")
    requested_action: str = Field(..., description="What they want done")
    can_self_serve: bool = Field(
        ..., description="Whether user can do it themselves"
    )
    self_serve_instructions: Optional[str] = Field(
        None, description="How to do it themselves"
    )
    requires_human: bool = Field(
        ..., description="Whether human intervention needed"
    )
    handoff_target: Optional[str] = Field(None, description="Who handles this")
    handoff_channel: Optional[str] = Field(None, description="How to submit request")
    expected_sla: Optional[str] = Field(None, description="Turnaround time")
    template_or_form: Optional[str] = Field(None, description="Link to form if any")


class SensitiveDataResponse(BaseModel):
    """Response for sensitive_data_request intent - PII/Legal/Medical/PCI requests."""

    intent: str = Field(default="sensitive_data_request", description="Intent type")
    request_declined: bool = Field(
        default=True, description="Always true - request is declined"
    )
    query_understood: str = Field(..., description="Paraphrase of what user asked")
    detected_categories: List[str] = Field(
        ..., description="Which categories: pii, legal, medical, pci"
    )
    applicable_policies: List[str] = Field(
        ..., description="Policy summaries that apply"
    )
    explanation: str = Field(..., description="Why the request was declined")
    alternative_suggestion: Optional[str] = Field(
        None, description="What user could do instead"
    )
    confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.HIGH,
        description="Always high for policy enforcement",
    )


class OutOfScopeResponse(BaseModel):
    """Response for out_of_scope intent - unrelated queries."""

    intent: str = Field(default="out_of_scope", description="Intent type")
    query_understood: str = Field(..., description="What was understood")
    reason_out_of_scope: str = Field(..., description="Why this can't be answered")
    closest_supported_topic: Optional[str] = Field(
        None, description="Nearest topic that could be helped with"
    )
    suggested_resource: Optional[str] = Field(
        None, description="Where they might find help"
    )
    suggested_queries: List[str] = Field(
        default_factory=list, description="Example in-scope queries"
    )


class ClarificationResponse(BaseModel):
    """Response when retrieval returns insufficient context."""

    intent: str = Field(default="clarification_needed", description="Intent type")
    query_understood: str = Field(..., description="What was understood")
    reason: str = Field(..., description="Why clarification is needed")
    suggestions: List[str] = Field(
        default_factory=list, description="Suggested ways to rephrase"
    )
    partial_answer: Optional[str] = Field(
        None, description="Partial answer if any context was found"
    )


# Type alias for all response types
IntentResponse = Union[
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
]


@dataclass
class ConversationTurn:
    """Single turn in a conversation history."""

    role: str  # "user" or "assistant"
    content: str
