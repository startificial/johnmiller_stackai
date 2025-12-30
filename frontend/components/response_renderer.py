"""
Intent-specific response rendering components.
"""

import json
from typing import Any, Optional

import streamlit as st


def render_response(content: str, intent: Optional[str] = None):
    """
    Route to appropriate renderer based on intent.

    Args:
        content: Response content (may be JSON string)
        intent: Intent classification from API (fallback)
    """
    # Try to parse JSON content
    data = _parse_content(content)

    # Use intent from parsed JSON if available, otherwise use API intent
    # The LLM response JSON contains the actual intent which may differ from API's classification
    actual_intent = data.get("intent", intent)

    # Route to specific renderer
    renderers = {
        "lookup": render_lookup_response,
        "explain": render_explain_response,
        "procedural": render_procedural_response,
        "troubleshoot": render_troubleshoot_response,
        "compare": render_compare_response,
        "status": render_status_response,
        "discovery": render_discovery_response,
        "contact": render_contact_response,
        "action": render_action_response,
        "out_of_scope": render_out_of_scope_response,
        "clarification_needed": render_clarification_response,
        "sensitive_data_request": render_sensitive_data_response,
        "hallucination_blocked": render_hallucination_blocked_response,
    }

    renderer = renderers.get(actual_intent, render_generic_response)
    renderer(data)


def _parse_content(content: str) -> dict[str, Any]:
    """Parse JSON content or return as raw."""
    if not content or not content.strip():
        return {"raw_content": "(Empty response)", "_empty": True}
    try:
        parsed = json.loads(content)
        if not parsed:
            return {"raw_content": "(Empty JSON object)", "_empty": True}
        return parsed
    except (json.JSONDecodeError, TypeError):
        return {"raw_content": content}


def render_generic_response(data: dict[str, Any]):
    """Fallback renderer for unknown intents or plain text."""
    # Check for empty response flag
    if data.get("_empty"):
        st.warning(data.get("raw_content", "No response received"))
        return

    # Check for raw content first
    if "raw_content" in data:
        st.markdown(data["raw_content"])
        return

    # Try common content fields
    for field in ["explanation", "summary", "content", "answer", "response"]:
        if content := data.get(field):
            st.markdown(content)
            return

    # Show key points if available
    if key_points := data.get("key_points"):
        for point in key_points:
            st.markdown(f"- {point}")
        return

    # Fallback: show nicely formatted JSON
    st.json(data)


def render_lookup_response(data: dict[str, Any]):
    """Render lookup/resource finding response."""
    found = data.get("found", False)

    if found:
        title = data.get("resource_title", "Resource Found")
        st.markdown(f"### {title}")

        if resource_type := data.get("resource_type"):
            st.caption(f"Type: {resource_type}")

        if summary := data.get("summary"):
            st.markdown(summary)

        if url := data.get("resource_url"):
            st.markdown(f"[View Resource]({url})")

        if related := data.get("related_resources"):
            st.markdown("**Related Resources:**")
            for r in related:
                st.markdown(f"- {r.get('title', r.get('name', 'Unknown'))}")
    else:
        st.warning("Resource not found")
        if suggestion := data.get("fallback_suggestion"):
            st.info(suggestion)


def render_explain_response(data: dict[str, Any]):
    """Render explanation with key points."""
    concept = data.get("concept", "Explanation")
    st.markdown(f"### {concept}")

    if explanation := data.get("explanation"):
        st.markdown(explanation)

    if key_points := data.get("key_points"):
        st.markdown("**Key Points:**")
        for point in key_points:
            st.markdown(f"- {point}")

    if analogy := data.get("analogy"):
        st.info(f"**Think of it like:** {analogy}")

    if related := data.get("related_concepts"):
        st.caption(f"Related: {', '.join(related)}")


def render_procedural_response(data: dict[str, Any]):
    """Render step-by-step instructions."""
    task = data.get("task", "Instructions")
    st.markdown(f"### {task}")

    if prereqs := data.get("prerequisites"):
        with st.expander("Prerequisites", expanded=False):
            for prereq in prereqs:
                st.markdown(f"- {prereq}")

    if time_est := data.get("estimated_time"):
        st.caption(f"Estimated time: {time_est}")

    st.markdown("**Steps:**")
    for step in data.get("steps", []):
        step_num = step.get("step_number", "")
        action = step.get("action", "")
        st.markdown(f"**{step_num}.** {action}")

        if details := step.get("details"):
            st.caption(details)

        if warning := step.get("warning"):
            st.warning(warning)

        if code := step.get("code_snippet"):
            st.code(code)

    if outcome := data.get("outcome"):
        st.success(f"**Expected Outcome:** {outcome}")

    if errors := data.get("common_errors"):
        with st.expander("Common Errors", expanded=False):
            for error in errors:
                st.markdown(f"- {error}")


def render_troubleshoot_response(data: dict[str, Any]):
    """Render troubleshooting with solutions."""
    problem = data.get("problem_summary", "Troubleshooting")
    st.markdown(f"### {problem}")

    if symptoms := data.get("symptoms_identified"):
        st.markdown("**Symptoms:**")
        for symptom in symptoms:
            st.markdown(f"- {symptom}")

    if causes := data.get("probable_causes"):
        st.markdown("**Probable Causes:**")
        for cause in causes:
            st.markdown(f"- {cause}")

    st.markdown("**Solutions:**")
    for solution in data.get("solutions", []):
        likelihood = solution.get("likelihood", "possible")
        icon = "likely" if likelihood == "likely" else ""
        expanded = likelihood == "likely"

        with st.expander(f"{icon} {solution.get('description', '')}", expanded=expanded):
            for step in solution.get("steps", []):
                st.markdown(f"- {step}")

    if escalation := data.get("escalation_path"):
        st.info(f"**Escalation:** {escalation}")


def render_compare_response(data: dict[str, Any]):
    """Render comparison table."""
    topic = data.get("comparison_topic", "Comparison")
    st.markdown(f"### {topic}")

    items = data.get("items", [])
    if items:
        # Show summary table
        for item in items:
            with st.expander(item.get("name", "Option"), expanded=True):
                st.markdown(item.get("summary", ""))
                st.markdown(f"**Best for:** {item.get('best_for', 'N/A')}")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Pros:**")
                    for pro in item.get("pros", []):
                        st.markdown(f"+ {pro}")
                with col2:
                    st.markdown("**Cons:**")
                    for con in item.get("cons", []):
                        st.markdown(f"- {con}")

    if rec := data.get("recommendation"):
        st.success(f"**Recommendation:** {rec}")

    if factors := data.get("decision_factors"):
        st.caption(f"Key factors: {', '.join(factors)}")


def render_status_response(data: dict[str, Any]):
    """Render status information."""
    subject = data.get("subject", "Status")
    st.markdown(f"### {subject}")

    status = data.get("current_status", "Unknown")
    status_type = data.get("status_type", "unknown")

    # Status badge
    status_colors = {
        "active": "green",
        "completed": "blue",
        "blocked": "red",
        "deprecated": "gray",
    }
    st.markdown(f"**Status:** {status} ({status_type})")

    if data.get("staleness_warning"):
        st.warning("This information may be outdated")

    if last_updated := data.get("last_updated"):
        st.caption(f"Last updated: {last_updated}")

    if changes := data.get("recent_changes"):
        st.markdown("**Recent Changes:**")
        for change in changes:
            st.markdown(f"- {change}")

    if owner := data.get("owner"):
        st.caption(f"Owner: {owner}")


def render_discovery_response(data: dict[str, Any]):
    """Render exploration/discovery results."""
    area = data.get("exploration_area", "Available Options")
    st.markdown(f"### {area}")

    if note := data.get("coverage_note"):
        st.caption(note)

    items = data.get("items_found", [])
    if items:
        for item in items:
            with st.expander(item.get("name", "Item"), expanded=False):
                st.markdown(item.get("description", ""))
                if link := item.get("link"):
                    st.markdown(f"[Learn more]({link})")

    if categories := data.get("categories"):
        st.markdown(f"**Categories:** {', '.join(categories)}")

    if follow_ups := data.get("follow_up_queries"):
        st.markdown("**Try asking:**")
        for query in follow_ups:
            st.markdown(f"- {query}")


def render_contact_response(data: dict[str, Any]):
    """Render contact information."""
    subject = data.get("subject", "Contact Information")
    st.markdown(f"### {subject}")

    if primary := data.get("primary_contact"):
        st.markdown("**Primary Contact:**")
        _render_contact(primary)

    if alts := data.get("alternative_contacts"):
        with st.expander("Alternative Contacts"):
            for contact in alts:
                _render_contact(contact)
                st.divider()

    if method := data.get("contact_method"):
        st.caption(f"Best contact method: {method}")

    if hours := data.get("office_hours"):
        st.caption(f"Hours: {hours}")


def _render_contact(contact: dict):
    """Render single contact details."""
    st.markdown(f"**{contact.get('name', 'Unknown')}**")
    if role := contact.get("role"):
        st.caption(role)
    if team := contact.get("team"):
        st.caption(f"Team: {team}")
    if email := contact.get("email"):
        st.markdown(f"Email: {email}")
    if slack := contact.get("slack_channel"):
        st.markdown(f"Slack: {slack}")


def render_action_response(data: dict[str, Any]):
    """Render action request response."""
    action = data.get("requested_action", "Action Request")
    st.markdown(f"### {action}")

    if data.get("can_self_serve"):
        st.success("You can do this yourself!")
        if instructions := data.get("self_serve_instructions"):
            st.markdown(instructions)
    elif data.get("requires_human"):
        st.info("This requires human assistance")
        if target := data.get("handoff_target"):
            st.markdown(f"**Contact:** {target}")
        if channel := data.get("handoff_channel"):
            st.markdown(f"**Via:** {channel}")
        if sla := data.get("expected_sla"):
            st.caption(f"Expected response: {sla}")

    if template := data.get("template_or_form"):
        st.markdown(f"**Form/Template:** {template}")


def render_out_of_scope_response(data: dict[str, Any]):
    """Render out of scope response."""
    st.warning("This question is outside the knowledge base scope")

    if reason := data.get("reason_out_of_scope"):
        st.markdown(reason)

    if closest := data.get("closest_supported_topic"):
        st.info(f"**Related topic:** {closest}")

    if resource := data.get("suggested_resource"):
        st.markdown(f"**Try:** {resource}")

    if suggestions := data.get("suggested_queries"):
        st.markdown("**You might ask:**")
        for query in suggestions:
            st.markdown(f"- {query}")


def render_clarification_response(data: dict[str, Any]):
    """Render clarification needed response."""
    st.info("I need more information to answer this question")

    if reason := data.get("reason"):
        st.markdown(reason)

    if partial := data.get("partial_answer"):
        st.markdown("**What I found:**")
        st.markdown(partial)

    if suggestions := data.get("suggestions"):
        st.markdown("**Try rephrasing:**")
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")


def render_sensitive_data_response(data: dict[str, Any]):
    """Render sensitive data request decline."""
    st.error("This request involves sensitive data and cannot be fulfilled")

    if categories := data.get("detected_categories"):
        st.markdown(f"**Categories:** {', '.join(categories)}")

    if policies := data.get("applicable_policies"):
        with st.expander("Applicable Policies"):
            for policy in policies:
                st.markdown(f"- {policy}")

    if explanation := data.get("explanation"):
        st.markdown(explanation)

    if suggestion := data.get("alternative_suggestion"):
        st.info(f"**Alternative:** {suggestion}")


def render_hallucination_blocked_response(data: dict[str, Any]):
    """Render when response was blocked due to hallucination."""
    st.warning("Response could not be verified against source documents")

    if reason := data.get("reason"):
        st.markdown(reason)

    if claims := data.get("unsupported_claims"):
        with st.expander("Unsupported Claims"):
            for claim in claims:
                st.markdown(f"- {claim.get('claim', claim)}")

    if question := data.get("clarifying_question"):
        st.info(f"**Suggestion:** {question}")

    if suggestion := data.get("fallback_suggestion"):
        st.markdown(f"**Try:** {suggestion}")
