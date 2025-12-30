"""
LLM Response Prompts

Intent-specific prompts for structured response generation.
Prompts are designed to be clear, maintainable, and produce valid JSON output.
"""

from backend.app.services.intent_classifier import Intent


# Shared citation instructions appended to all KB-dependent prompts
CITATION_INSTRUCTIONS = """
## Citation Requirements

You MUST cite sources for every factual claim. Follow these rules:

1. **Reference sources by ID**: Use source IDs (src_0, src_1, etc.) in your citations.

2. **Citation types**:
   - "direct": Source explicitly states this
   - "paraphrased": You rephrased what the source says
   - "inferred": Reasonable conclusion from source (use sparingly)

3. **No source, no claim**: If sources don't support an answer, say so honestly.

4. **Conflicting sources**: If sources disagree, note the conflict.
"""


# System prompt for all responses
SYSTEM_PROMPT = """You are a knowledgeable RAG assistant that provides accurate, well-cited responses based on retrieved context. Always respond with valid JSON matching the specified schema."""


LOOKUP_PROMPT = """You are helping the user find a specific resource they believe exists.

{citation_instructions}

## Your Task
1. Identify if the requested resource was found in the provided context
2. If found, provide the location/URL and a brief summary
3. If not found, acknowledge and suggest related resources

## Retrieved Context
{context}

{conversation_context}

## User Query
{query}

## Response Format (JSON)
Respond with valid JSON matching LookupResponse:
- found: boolean
- resource_title: string or null
- resource_url: string or null
- resource_type: "doc"|"page"|"file"|"tool"|"api"|null
- summary: string describing what was found or not
- related_resources: list of {{"title": string, "source_id": string}}
- confidence: "high"|"medium"|"low"|"uncertain"
- query_understood: string (paraphrase of query)
- sources: list of source objects used
- citations: list of citation objects
- needs_clarification: boolean
- clarifying_question: string or null"""


EXPLAIN_PROMPT = """You are explaining a concept clearly and accurately.

{citation_instructions}

## Your Task
1. Provide a clear, accessible explanation based on the sources
2. Highlight key points for understanding
3. Mention related concepts that may help
4. Use appropriate technical depth for the audience

## Retrieved Context
{context}

{conversation_context}

## User Query
{query}

## Response Format (JSON)
Respond with valid JSON matching ExplainResponse:
- concept: string (the concept being explained)
- explanation: string (clear explanation)
- key_points: list of strings
- analogy: string or null (optional illustration)
- related_concepts: list of strings
- technical_depth: "basic"|"intermediate"|"advanced"
- confidence: "high"|"medium"|"low"|"uncertain"
- query_understood: string
- sources: list of source objects
- citations: list of citation objects
- needs_clarification: boolean
- clarifying_question: string or null"""


PROCEDURAL_PROMPT = """You are providing step-by-step instructions for a task.

{citation_instructions}

## Your Task
1. Provide clear, sequential steps from the sources
2. Include prerequisites and warnings where appropriate
3. Describe what success looks like
4. Note common errors to avoid

## Retrieved Context
{context}

{conversation_context}

## User Query
{query}

## Response Format (JSON)
Respond with valid JSON matching ProcedureResponse:
- task: string (the task being explained)
- prerequisites: list of strings
- steps: list of {{"step_number": int, "action": string, "details": string|null, "warning": string|null, "code_snippet": string|null}}
- estimated_time: string or null
- outcome: string (what success looks like)
- common_errors: list of strings
- next_steps: list of strings
- confidence: "high"|"medium"|"low"|"uncertain"
- query_understood: string
- sources: list of source objects
- citations: list of citation objects
- needs_clarification: boolean
- clarifying_question: string or null"""


TROUBLESHOOT_PROMPT = """You are helping diagnose and resolve a problem.

{citation_instructions}

## Your Task
1. Summarize the problem based on user description
2. Identify symptoms mentioned
3. List probable causes from the sources
4. Provide solutions ordered by likelihood
5. Include escalation path if needed

## Retrieved Context
{context}

{conversation_context}

## User Query
{query}

## Response Format (JSON)
Respond with valid JSON matching TroubleshootResponse:
- problem_summary: string (one-line description)
- symptoms_identified: list of strings
- diagnostic_questions: list of strings (to narrow down)
- probable_causes: list of strings
- solutions: list of {{"description": string, "steps": list of strings, "likelihood": "likely"|"possible"|"rare"}}
- escalation_path: string or null
- related_issues: list of strings
- confidence: "high"|"medium"|"low"|"uncertain"
- query_understood: string
- sources: list of source objects
- citations: list of citation objects
- needs_clarification: boolean
- clarifying_question: string or null"""


COMPARE_PROMPT = """You are comparing options to help the user make a decision.

{citation_instructions}

## Your Task
1. Identify the items being compared
2. Compare across relevant dimensions from the sources
3. List pros and cons for each option
4. Provide a recommendation only if sources clearly support it

## Retrieved Context
{context}

{conversation_context}

## User Query
{query}

## Response Format (JSON)
Respond with valid JSON matching CompareResponse:
- comparison_topic: string
- items: list of {{"name": string, "summary": string, "pros": list, "cons": list, "best_for": string}}
- dimensions: list of strings (comparison criteria)
- comparison_table: dict of {{dimension: {{item: value}}}}
- recommendation: string or null
- decision_factors: list of strings (questions to help decide)
- confidence: "high"|"medium"|"low"|"uncertain"
- query_understood: string
- sources: list of source objects
- citations: list of citation objects
- needs_clarification: boolean
- clarifying_question: string or null"""


STATUS_PROMPT = """You are providing current status information.

{citation_instructions}

## Your Task
1. Report the current status from the sources
2. Note when the information was last updated
3. Assess source freshness (current/recent/stale)
4. Include recent changes if documented
5. Flag if live data may be needed

## Retrieved Context
{context}

## Current Date
{current_date}

{conversation_context}

## User Query
{query}

## Response Format (JSON)
Respond with valid JSON matching StatusResponse:
- subject: string (what is being reported on)
- current_status: string
- status_type: "active"|"completed"|"blocked"|"deprecated"|"unknown"
- last_updated: string or null
- source_freshness: "current"|"recent"|"stale"|"unknown"
- recent_changes: list of strings
- next_expected_update: string or null
- owner: string or null
- staleness_warning: boolean
- confidence: "high"|"medium"|"low"|"uncertain"
- query_understood: string
- sources: list of source objects
- citations: list of citation objects
- needs_clarification: boolean
- clarifying_question: string or null"""


DISCOVERY_PROMPT = """You are helping the user explore what's available.

{citation_instructions}

## Your Task
1. List available options/items from the sources
2. Group into categories if appropriate
3. Suggest ways to narrow down the exploration
4. Note how complete the list is

## Retrieved Context
{context}

{conversation_context}

## User Query
{query}

## Response Format (JSON)
Respond with valid JSON matching DiscoveryResponse:
- exploration_area: string
- items_found: list of {{"name": string, "description": string, "relevance": string, "link": string|null}}
- categories: list of strings
- suggested_filters: list of strings
- follow_up_queries: list of strings
- coverage_note: string (how complete this list is)
- confidence: "high"|"medium"|"low"|"uncertain"
- query_understood: string
- sources: list of source objects
- citations: list of citation objects
- needs_clarification: boolean
- clarifying_question: string or null"""


CONTACT_PROMPT = """You are helping find the right person or team to contact.

{citation_instructions}

## Your Task
1. Identify contacts from the sources (don't guess)
2. Note the data source (org chart, docs, etc.)
3. Include escalation path if available
4. If no contact found, explain how to find the right person

## Retrieved Context
{context}

{conversation_context}

## User Query
{query}

## Response Format (JSON)
Respond with valid JSON matching ContactResponse:
- subject: string (what/who is being looked up)
- primary_contact: {{"name": string, "role": string|null, "team": string|null, "email": string|null, "slack_channel": string|null, "relationship_to_query": string}} or null
- alternative_contacts: list of contact objects
- escalation_path: string or null
- contact_method: string (recommended way to reach out)
- office_hours: string or null
- data_source: "org_chart"|"ownership_registry"|"documentation"|"inferred"
- confidence: "high"|"medium"|"low"|"uncertain"
- query_understood: string
- sources: list of source objects
- citations: list of citation objects
- needs_clarification: boolean
- clarifying_question: string or null"""


ACTION_PROMPT = """You are helping the user get something done.

{citation_instructions}

## Your Task
1. Understand what action is being requested
2. Determine if user can self-serve or needs human help
3. Provide instructions if self-service is possible
4. Identify who handles this if human help is needed

## Retrieved Context
{context}

{conversation_context}

## User Query
{query}

## Response Format (JSON)
Respond with valid JSON matching ActionResponse:
- requested_action: string
- can_self_serve: boolean
- self_serve_instructions: string or null
- requires_human: boolean
- handoff_target: string or null
- handoff_channel: string or null
- expected_sla: string or null
- template_or_form: string or null
- confidence: "high"|"medium"|"low"|"uncertain"
- query_understood: string
- sources: list of source objects
- citations: list of citation objects
- needs_clarification: boolean
- clarifying_question: string or null"""


OUT_OF_SCOPE_PROMPT = """You are explaining that the query is outside the knowledge base scope.

## Your Task
1. Acknowledge what the user is asking about
2. Politely explain why you can't help with this specific query
3. Suggest what you CAN help with
4. Offer alternative resources if appropriate

## User Query
{query}

{conversation_context}

## Response Format (JSON)
Respond with valid JSON matching OutOfScopeResponse:
- query_understood: string (what was understood)
- reason_out_of_scope: string (why this can't be answered)
- closest_supported_topic: string or null
- suggested_resource: string or null
- suggested_queries: list of example in-scope queries"""


CLARIFICATION_PROMPT = """You need to ask for clarification because the retrieved context is insufficient.

## Your Task
1. Acknowledge what you understood from the query
2. Explain why you need more information
3. Suggest ways the user could rephrase or provide more detail
4. Offer any partial answer if some context was relevant

## Retrieved Context
{context}

## User Query
{query}

{conversation_context}

## Response Format (JSON)
Respond with valid JSON matching ClarificationResponse:
- query_understood: string
- reason: string (why clarification is needed)
- suggestions: list of strings (ways to rephrase)
- partial_answer: string or null"""


# Prompt registry for easy lookup by intent
INTENT_PROMPTS = {
    Intent.LOOKUP: LOOKUP_PROMPT,
    Intent.EXPLAIN: EXPLAIN_PROMPT,
    Intent.PROCEDURAL: PROCEDURAL_PROMPT,
    Intent.TROUBLESHOOT: TROUBLESHOOT_PROMPT,
    Intent.COMPARE: COMPARE_PROMPT,
    Intent.STATUS: STATUS_PROMPT,
    Intent.DISCOVERY: DISCOVERY_PROMPT,
    Intent.CONTACT: CONTACT_PROMPT,
    Intent.ACTION: ACTION_PROMPT,
    Intent.OUT_OF_SCOPE: OUT_OF_SCOPE_PROMPT,
}

# Intents that require knowledge base retrieval
KB_DEPENDENT_INTENTS = {
    Intent.LOOKUP,
    Intent.EXPLAIN,
    Intent.PROCEDURAL,
    Intent.TROUBLESHOOT,
    Intent.COMPARE,
    Intent.STATUS,
    Intent.DISCOVERY,
    Intent.CONTACT,
    Intent.ACTION,
}


def get_prompt_for_intent(intent: Intent) -> str:
    """
    Get the appropriate prompt template for an intent.

    Args:
        intent: The classified intent

    Returns:
        Prompt template string
    """
    return INTENT_PROMPTS.get(intent, OUT_OF_SCOPE_PROMPT)


def requires_kb_retrieval(intent: Intent) -> bool:
    """
    Check if an intent requires knowledge base retrieval.

    Args:
        intent: The classified intent

    Returns:
        True if RAG retrieval is needed, False otherwise
    """
    return intent in KB_DEPENDENT_INTENTS
