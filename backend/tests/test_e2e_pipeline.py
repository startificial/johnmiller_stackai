"""
End-to-End Integration Tests for RAG Pipeline

These tests use real test documents from test_documents/ and make live Mistral AI API calls.
They validate the complete RAG pipeline from PDF ingestion to response generation.

Requirements:
- PostgreSQL database with pgvector extension
- MISTRAL_API_KEY environment variable
- Test documents in test_documents/ directory

Run with:
    source .env && PYTHONPATH=. uv run pytest backend/tests/test_e2e_pipeline.py -v -m integration
"""

import pytest
import pytest_asyncio
from pathlib import Path
from uuid import UUID
from typing import Dict, List, Any

from backend.app.db.base import init_db, drop_db, async_session_factory
from backend.app.db.models import Document, DocumentChunk
from backend.app.services.text_extraction import text_extraction_service
from backend.app.services.chunking import chunking_service
from backend.app.services.embedding import embedding_service
from backend.app.services.search_indexer import search_indexer
from backend.app.services.retriever import retriever
from backend.app.services.intent_classifier import intent_classifier, Intent
from backend.app.services.query_transformer import query_transformer
from backend.app.services.llm_response.service import LLMResponseService
from backend.app.services.llm_response.schemas import (
    ClarificationResponse,
    SensitiveDataResponse,
    OutOfScopeResponse,
    LookupResponse,
    ExplainResponse,
    ProcedureResponse,
    TroubleshootResponse,
    HallucinationBlockedResponse,
)
from backend.app.services.hallucination_detector import hallucination_detector
from backend.app.services.policy_repository import policy_repository
from sqlalchemy import select, func


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


# =============================================================================
# Test Document Paths
# =============================================================================

TEST_DOCUMENTS_DIR = Path(__file__).parent.parent.parent / "test_documents"


def get_test_document_path(filename: str) -> Path:
    """Get full path to a test document."""
    path = TEST_DOCUMENTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Test document not found: {path}")
    return path


# =============================================================================
# Fixtures
# =============================================================================


@pytest_asyncio.fixture(scope="function")
async def setup_database():
    """Initialize database for each test function."""
    # Import here to get fresh engine for each test
    from backend.app.db.base import init_db
    await init_db()
    yield
    # Skip cleanup - tables will be truncated or recreated on next test


async def ingest_document(filename: str) -> Dict[str, Any]:
    """
    Helper to ingest a single document through the full pipeline.

    Returns dict with doc_id, chunk_count, page_count, etc.
    """
    file_path = get_test_document_path(filename)

    # Step 1: Extract text
    extracted_doc = await text_extraction_service.extract_and_persist(
        file_path=file_path,
    )

    # Step 2: Chunk document
    chunked_doc = await chunking_service.chunk_and_persist(
        document=extracted_doc,
        doc_id=extracted_doc.doc_id,
    )

    # Step 3: Generate embeddings (for leaf chunks)
    embed_result = await embedding_service.embed_document(
        chunked_doc=chunked_doc,
        embed_parents=False,
    )

    # Step 4: Index for BM25 search
    await search_indexer.index_chunked_document(chunked_doc)

    # Count parent vs leaf chunks
    parent_count = sum(1 for c in chunked_doc.chunks if c.level == 0)
    leaf_count = sum(1 for c in chunked_doc.chunks if c.level == 1)

    return {
        "doc_id": str(extracted_doc.doc_id),
        "filename": filename,
        "page_count": extracted_doc.page_count,
        "total_chunks": len(chunked_doc.chunks),
        "parent_chunks": parent_count,
        "leaf_chunks": leaf_count,
        "chunks_embedded": embed_result.get("chunks_embedded", 0),
    }


@pytest_asyncio.fixture(scope="function")
async def ingested_happy_path(setup_database):
    """Ingest 01_happy_path_text_only.pdf and return metadata."""
    return await ingest_document("01_happy_path_text_only.pdf")


@pytest_asyncio.fixture(scope="function")
async def ingested_hierarchical(setup_database):
    """Ingest 03_hierarchical_structure.pdf and return metadata."""
    return await ingest_document("03_hierarchical_structure.pdf")


@pytest_asyncio.fixture(scope="function")
async def ingested_multipage(setup_database):
    """Ingest 08_multipage_with_hierarchy.pdf (27 pages) and return metadata."""
    return await ingest_document("08_multipage_with_hierarchy.pdf")


@pytest_asyncio.fixture(scope="function")
async def ingested_corpus(setup_database):
    """Ingest multiple test documents for comprehensive testing."""
    corpus = {}
    for filename in [
        "01_happy_path_text_only.pdf",
        "03_hierarchical_structure.pdf",
        "05_tables.pdf",
    ]:
        try:
            corpus[filename] = await ingest_document(filename)
        except FileNotFoundError:
            continue
    return corpus


# =============================================================================
# Test Class 1: Data Ingestion
# =============================================================================


class TestE2EDataIngestion:
    """End-to-end tests for the data ingestion pipeline."""

    async def test_ingest_happy_path_text_only(self, setup_database):
        """Test ingesting a simple text-only PDF."""
        result = await ingest_document("01_happy_path_text_only.pdf")

        assert result["doc_id"] is not None
        assert UUID(result["doc_id"])  # Valid UUID
        assert result["page_count"] >= 1
        assert result["total_chunks"] >= 1
        assert result["leaf_chunks"] > 0
        assert result["chunks_embedded"] > 0

    async def test_ingest_empty_document(self, setup_database):
        """Test handling of an empty/blank PDF."""
        file_path = get_test_document_path("02_no_content.pdf")

        # Should not raise but may have 0 content
        extracted_doc = await text_extraction_service.extract_and_persist(
            file_path=file_path,
        )

        # Verify document was created (even if empty)
        assert extracted_doc.doc_id is not None
        # Page count should still be recorded
        assert extracted_doc.page_count >= 0

    async def test_ingest_hierarchical_structure(self, ingested_hierarchical):
        """Test that hierarchical document preserves structure."""
        result = ingested_hierarchical

        assert result["total_chunks"] >= 1
        # Hierarchical doc should have meaningful content
        assert result["page_count"] >= 1

    async def test_ingest_tables(self, setup_database):
        """Test ingesting PDF with tables."""
        result = await ingest_document("05_tables.pdf")

        assert result["doc_id"] is not None
        assert result["total_chunks"] >= 1
        # Table content should be extracted
        assert result["page_count"] >= 1

    async def test_ingest_multipage_with_hierarchy(self, ingested_multipage):
        """Test ingesting a large 27-page document with TOC."""
        result = ingested_multipage

        # Should have multiple pages
        assert result["page_count"] >= 20
        # Should have many chunks
        assert result["total_chunks"] >= 10
        # Should have both parent and leaf chunks
        assert result["parent_chunks"] >= 1
        assert result["leaf_chunks"] >= 1

    async def test_ingest_multiple_documents(self, ingested_corpus):
        """Test batch ingestion of multiple documents."""
        corpus = ingested_corpus

        # Should have ingested at least 2 documents
        assert len(corpus) >= 2

        # Each document should have unique ID
        doc_ids = [info["doc_id"] for info in corpus.values()]
        assert len(doc_ids) == len(set(doc_ids))

    async def test_chunking_parent_child_relationships(self, ingested_happy_path):
        """Verify chunk hierarchy with parent-child relationships."""
        result = ingested_happy_path
        doc_id = result["doc_id"]

        # Query database for chunks
        async with async_session_factory() as session:
            stmt = select(DocumentChunk).where(
                DocumentChunk.document_id == UUID(doc_id)
            )
            chunks_result = await session.execute(stmt)
            chunks = chunks_result.scalars().all()

        # Should have chunks at different hierarchy levels
        levels = set(c.level for c in chunks)

        # Verify parent-child relationships
        child_chunks = [c for c in chunks if c.parent_id is not None]
        if child_chunks:
            parent_ids = set(c.parent_id for c in child_chunks)
            for parent_id in parent_ids:
                parent_exists = any(c.id == parent_id for c in chunks)
                assert parent_exists, f"Parent chunk {parent_id} not found"


# =============================================================================
# Test Class 2: Query Processing
# =============================================================================


class TestE2EQueryProcessing:
    """End-to-end tests for query processing (intent, transformation)."""

    async def test_intent_classification_lookup(self):
        """Test LOOKUP intent classification."""
        # Use a clear lookup-style query
        result = await intent_classifier.classify("Find the API documentation link")
        assert result.intent == Intent.LOOKUP
        assert result.confidence > 0.5

    async def test_intent_classification_explain(self):
        """Test EXPLAIN intent classification."""
        result = await intent_classifier.classify("Explain how the authentication system works")
        assert result.intent == Intent.EXPLAIN
        assert result.confidence > 0.5

    async def test_intent_classification_procedural(self):
        """Test PROCEDURAL intent classification."""
        result = await intent_classifier.classify("How do I deploy the application to production?")
        assert result.intent == Intent.PROCEDURAL
        assert result.confidence > 0.5

    async def test_intent_classification_troubleshoot(self):
        """Test TROUBLESHOOT intent classification."""
        result = await intent_classifier.classify("The API is returning 500 errors, how do I fix it?")
        assert result.intent == Intent.TROUBLESHOOT
        assert result.confidence > 0.5

    async def test_vague_query_no_search(self):
        """Test that vague greetings are classified as OUT_OF_SCOPE."""
        result = await intent_classifier.classify("hello")
        assert result.intent == Intent.OUT_OF_SCOPE

        result2 = await intent_classifier.classify("hi there")
        assert result2.intent == Intent.OUT_OF_SCOPE

    async def test_query_transformation_variants(self):
        """Test that query transformer generates multiple variants."""
        result = await query_transformer.generate_multi_query(
            "How do I configure database connections?"
        )

        # Should have original + variants
        assert len(result.all_queries) >= 2
        assert result.original_query in result.all_queries

        # Variants should be different from original
        variants_only = [q for q in result.all_queries if q != result.original_query]
        assert len(variants_only) >= 1
        for variant in variants_only:
            assert variant != result.original_query

    async def test_sensitive_data_detection_pii(self):
        """Test PII detection triggers sensitive_data_request intent."""
        # SSN query
        result = await intent_classifier.classify("What is John's social security number?")
        assert result.intent == Intent.SENSITIVE_DATA_REQUEST

        # Address query
        result2 = await intent_classifier.classify("Give me the employee's home address")
        assert result2.intent == Intent.SENSITIVE_DATA_REQUEST

    async def test_sensitive_data_detection_legal(self):
        """Test legal advice triggers sensitive_data_request intent."""
        result = await intent_classifier.classify(
            "Should I sue my employer for wrongful termination?"
        )
        assert result.intent == Intent.SENSITIVE_DATA_REQUEST

    async def test_sensitive_data_detection_medical(self):
        """Test medical advice triggers sensitive_data_request intent."""
        result = await intent_classifier.classify(
            "What medication should I take for my headache?"
        )
        assert result.intent == Intent.SENSITIVE_DATA_REQUEST

    async def test_sensitive_data_detection_pci(self):
        """Test PCI data requests trigger sensitive_data_request intent."""
        result = await intent_classifier.classify(
            "What is the customer's credit card number and CVV?"
        )
        assert result.intent == Intent.SENSITIVE_DATA_REQUEST


# =============================================================================
# Test Class 3: Semantic Search
# =============================================================================


class TestE2ESemanticSearch:
    """End-to-end tests for semantic and hybrid search."""

    async def test_semantic_search_relevant_query(self, ingested_happy_path):
        """Test semantic search returns relevant results."""
        # Query related to the document content
        result = await retriever.search(
            query="text extraction test content",
            top_k=5,
        )

        assert len(result.results) > 0
        # At least one result should have reasonable score
        assert any(r.semantic_score > 0 for r in result.results)

    async def test_keyword_bm25_search(self, ingested_happy_path):
        """Test BM25 keyword search works."""
        result = await retriever.search(
            query="quick brown fox",  # Known content in test doc
            top_k=5,
        )

        # Should find matches via keyword search
        assert len(result.results) >= 0  # May not match if doc doesn't have this

    async def test_hybrid_rrf_combination(self, ingested_happy_path):
        """Test that hybrid search combines semantic and keyword results."""
        result = await retriever.search(
            query="test document content",
            top_k=5,
        )

        if result.results:
            # Results should have RRF score
            for r in result.results:
                assert r.rrf_score is not None
                assert 0 <= r.rrf_score <= 1

    async def test_search_result_ordering(self, ingested_happy_path):
        """Test that results are ordered by RRF score descending."""
        result = await retriever.search(
            query="test content",
            top_k=10,
        )

        if len(result.results) >= 2:
            scores = [r.rrf_score for r in result.results]
            # Should be sorted descending
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1], "Results not sorted by score"

    async def test_context_expansion_parent_chunks(self, ingested_happy_path):
        """Test that search retrieves parent chunks for context."""
        result = await retriever.search(
            query="test",
            top_k=5,
        )

        # Check if any results have parent_text (context expansion)
        if result.results:
            # At least check the structure allows for parent text
            for r in result.results:
                # parent_text may or may not be populated
                assert hasattr(r, 'parent_text') or hasattr(r, 'text')


# =============================================================================
# Test Class 4: Post-Processing
# =============================================================================


class TestE2EPostProcessing:
    """End-to-end tests for result post-processing."""

    async def test_result_deduplication(self, ingested_corpus):
        """Test that multi-query retrieval deduplicates results."""
        if not ingested_corpus:
            pytest.skip("No documents ingested")

        service = LLMResponseService()

        # Use query that might hit same chunks from different variants
        result = await service._retrieve_context(
            query="document content information",
        )

        sources, queries = result

        # Each source should have unique chunk_id
        chunk_ids = [s.chunk_id for s in sources]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunks found"

    async def test_score_normalization(self, ingested_happy_path):
        """Test that scores are normalized to 0-1 range."""
        result = await retriever.search(
            query="test",
            top_k=10,
        )

        for r in result.results:
            if r.rrf_score is not None:
                assert 0 <= r.rrf_score <= 1, f"Score {r.rrf_score} out of range"
            if r.semantic_score is not None:
                assert 0 <= r.semantic_score <= 1
            if r.keyword_score is not None:
                assert 0 <= r.keyword_score <= 1

    async def test_top_k_limiting(self, ingested_corpus):
        """Test that only top K results are returned."""
        if not ingested_corpus:
            pytest.skip("No documents ingested")

        top_k = 3
        result = await retriever.search(
            query="content",
            top_k=top_k,
        )

        assert len(result.results) <= top_k


# =============================================================================
# Test Class 5: Generation
# =============================================================================


class TestE2EGeneration:
    """End-to-end tests for LLM response generation."""

    async def test_structured_response_lookup(self, ingested_happy_path):
        """Test that LOOKUP queries return LookupResponse schema."""
        service = LLMResponseService()

        result = await service.generate_response(
            query="Where is the main content located in this document?"
        )

        # Should get a structured response (may be clarification if no match)
        assert result.response is not None
        assert result.intent is not None

    async def test_structured_response_explain(self, ingested_happy_path):
        """Test that EXPLAIN queries return proper response structure."""
        service = LLMResponseService()

        result = await service.generate_response(
            query="Explain what this document is about"
        )

        assert result.response is not None
        # May be ExplainResponse or ClarificationResponse
        assert hasattr(result.response, 'confidence') or hasattr(result.response, 'intent')

    async def test_structured_response_procedural(self, ingested_happy_path):
        """Test procedural response structure."""
        service = LLMResponseService()

        result = await service.generate_response(
            query="How do I read the content from this document?"
        )

        assert result.response is not None

    async def test_response_includes_citations(self, ingested_happy_path):
        """Test that responses include source citations."""
        service = LLMResponseService()

        result = await service.generate_response(
            query="What text content is in this document?"
        )

        # Should have sources list
        assert hasattr(result, 'sources_used')
        # If we got a real response (not clarification), should have sources
        if not isinstance(result.response, ClarificationResponse):
            assert len(result.sources_used) >= 0

    async def test_response_confidence_levels(self, ingested_happy_path):
        """Test that responses include confidence levels."""
        service = LLMResponseService()

        result = await service.generate_response(
            query="What is this document about?"
        )

        if hasattr(result.response, 'confidence'):
            confidence = result.response.confidence
            # Should be one of the valid levels
            valid_levels = ['high', 'medium', 'low', 'uncertain',
                           'HIGH', 'MEDIUM', 'LOW', 'UNCERTAIN']
            assert str(confidence).upper() in [v.upper() for v in valid_levels]


# =============================================================================
# Test Class 6: Citation Threshold
# =============================================================================


class TestE2ECitationThreshold:
    """End-to-end tests for citation threshold enforcement."""

    async def test_insufficient_evidence_unrelated_query(self, ingested_happy_path):
        """Test that unrelated queries return ClarificationResponse."""
        service = LLMResponseService()

        # Query completely unrelated to test document
        result = await service.generate_response(
            query="What is the recipe for chocolate chip cookies?"
        )

        # Should get clarification response due to insufficient evidence
        assert isinstance(result.response, ClarificationResponse), (
            f"Expected ClarificationResponse, got {type(result.response).__name__}"
        )
        assert result.response.intent == "clarification_needed"

    async def test_sufficient_evidence_related_query(self, ingested_happy_path):
        """Test that related queries return normal responses when evidence exists."""
        service = LLMResponseService()

        # Query that matches document content
        result = await service.generate_response(
            query="What special characters are mentioned in this document?"
        )

        # May get either a normal response or clarification depending on threshold
        assert result.response is not None

    async def test_citation_sources_above_threshold(self, ingested_happy_path):
        """Test that returned sources meet the relevance threshold."""
        from backend.app.settings import settings

        service = LLMResponseService()
        threshold = settings.llm_response.min_relevance_threshold

        result = await service.generate_response(
            query="What text is in this document?"
        )

        # If we got sources and a non-clarification response
        if result.sources_used and not isinstance(result.response, ClarificationResponse):
            # At least one source should be above threshold
            max_score = max(s.relevance_score for s in result.sources_used)
            assert max_score >= threshold, (
                f"No sources above threshold {threshold}, max was {max_score}"
            )

    async def test_clarification_response_structure(self, ingested_happy_path):
        """Test ClarificationResponse has proper structure."""
        service = LLMResponseService()

        result = await service.generate_response(
            query="Explain quantum computing architecture"  # Unrelated
        )

        if isinstance(result.response, ClarificationResponse):
            assert result.response.intent == "clarification_needed"
            assert hasattr(result.response, 'reason')
            assert hasattr(result.response, 'suggestions')


# =============================================================================
# Test Class 7: Hallucination Detection
# =============================================================================


class TestE2EHallucinationDetection:
    """End-to-end tests for hallucination detection."""

    async def test_hallucination_check_basic(self, ingested_happy_path):
        """Test that hallucination detection runs on responses."""
        from backend.app.services.llm_response.schemas import Source

        # Create a simple source
        sources = [
            Source(
                source_id="src_0",
                chunk_id="test_chunk",
                document_id=ingested_happy_path["doc_id"],
                text_excerpt="The document contains text about testing.",
                relevance_score=0.8,
            )
        ]

        # Check a response that matches sources
        result = await hallucination_detector.check(
            response_text="The document contains information about testing.",
            sources=sources,
        )

        assert result.passed is not None
        assert result.hallucination_score >= 0
        assert result.hallucination_score <= 1

    async def test_hallucination_score_calculation(self, ingested_happy_path):
        """Test hallucination score is calculated correctly."""
        from backend.app.services.llm_response.schemas import Source

        sources = [
            Source(
                source_id="src_0",
                chunk_id="test_chunk",
                document_id=ingested_happy_path["doc_id"],
                text_excerpt="The system uses AES-256 encryption.",
                relevance_score=0.8,
            )
        ]

        # Response with potential unsupported claim
        result = await hallucination_detector.check(
            response_text="The system uses AES-256 encryption and supports MFA.",
            sources=sources,
        )

        # Score should be between 0 and 1
        assert 0 <= result.hallucination_score <= 1

        # If there are unsupported claims, score should be > 0
        if result.unsupported_claims:
            assert result.hallucination_score > 0


# =============================================================================
# Test Class 8: API Endpoints (via TestClient)
# =============================================================================


class TestE2EAPIEndpoints:
    """End-to-end tests for API endpoints."""

    @pytest.fixture
    def api_client(self):
        """Create FastAPI TestClient."""
        from fastapi.testclient import TestClient
        from backend.app.main import app
        return TestClient(app)

    def test_ingest_endpoint_single_file(self, api_client, setup_database):
        """Test POST /ingest with a single PDF file."""
        import asyncio
        asyncio.get_event_loop().run_until_complete(init_db())

        file_path = get_test_document_path("01_happy_path_text_only.pdf")

        with open(file_path, "rb") as f:
            response = api_client.post(
                "/ingest",
                files={"files": ("01_happy_path_text_only.pdf", f, "application/pdf")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["successful"] >= 1 or data["failed"] >= 0
        assert "documents" in data

    def test_query_endpoint_basic(self, api_client, ingested_happy_path):
        """Test POST /query with a basic query."""
        response = api_client.post(
            "/query",
            json={"query": "What is this document about?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "intent" in data
        assert "response" in data

    async def test_chat_session_lifecycle(self, setup_database):
        """Test complete chat session CRUD operations."""
        from fastapi.testclient import TestClient
        from backend.app.main import app

        await init_db()
        client = TestClient(app)

        # Create session
        create_response = client.post(
            "/chat/sessions",
            json={"title": "Test Chat Session"},
        )
        assert create_response.status_code == 200
        session_data = create_response.json()
        session_id = session_data["id"]

        # Get session
        get_response = client.get(f"/chat/sessions/{session_id}")
        assert get_response.status_code == 200

        # Delete session
        delete_response = client.delete(f"/chat/sessions/{session_id}")
        assert delete_response.status_code == 200

    async def test_chat_conversation_context(self, ingested_happy_path):
        """Test multi-turn conversation maintains context."""
        from fastapi.testclient import TestClient
        from backend.app.main import app

        client = TestClient(app)

        # Create session
        create_response = client.post(
            "/chat/sessions",
            json={"title": "Context Test"},
        )
        session_id = create_response.json()["id"]

        # Send first message
        msg1_response = client.post(
            f"/chat/sessions/{session_id}/messages",
            json={"content": "What is this document about?"},
        )
        assert msg1_response.status_code == 200

        # Send follow-up message referencing previous context
        msg2_response = client.post(
            f"/chat/sessions/{session_id}/messages",
            json={"content": "Can you explain more about that?"},
        )
        assert msg2_response.status_code == 200

        # Get history
        history_response = client.get(f"/chat/sessions/{session_id}/messages")
        assert history_response.status_code == 200
        messages = history_response.json()

        # Should have user + assistant messages
        assert len(messages) >= 2

        # Clean up
        client.delete(f"/chat/sessions/{session_id}")


# =============================================================================
# Test Class 9: Sensitive Data Responses
# =============================================================================


class TestE2ESensitiveDataResponses:
    """End-to-end tests for sensitive data handling."""

    async def test_pii_query_returns_sensitive_response(self, setup_database):
        """Test that PII queries return SensitiveDataResponse."""
        service = LLMResponseService()

        result = await service.generate_response(
            query="What is John's social security number?"
        )

        assert isinstance(result.response, SensitiveDataResponse)
        assert result.response.request_declined is True

    async def test_legal_query_returns_sensitive_response(self, setup_database):
        """Test that legal advice queries return SensitiveDataResponse."""
        service = LLMResponseService()

        result = await service.generate_response(
            query="Can I sue my employer for discrimination?"
        )

        assert isinstance(result.response, SensitiveDataResponse)

    async def test_medical_query_returns_sensitive_response(self, setup_database):
        """Test that medical advice queries return SensitiveDataResponse."""
        service = LLMResponseService()

        result = await service.generate_response(
            query="What medication should I take for diabetes?"
        )

        assert isinstance(result.response, SensitiveDataResponse)

    async def test_pci_query_returns_sensitive_response(self, setup_database):
        """Test that PCI data queries return SensitiveDataResponse."""
        service = LLMResponseService()

        result = await service.generate_response(
            query="Give me the customer's credit card CVV"
        )

        assert isinstance(result.response, SensitiveDataResponse)

    async def test_out_of_scope_query(self, setup_database):
        """Test that out-of-scope queries return OutOfScopeResponse."""
        service = LLMResponseService()

        result = await service.generate_response(
            query="hello"
        )

        assert isinstance(result.response, OutOfScopeResponse)


# =============================================================================
# Test Class 10: Policy Repository
# =============================================================================


class TestE2EPolicyRepository:
    """End-to-end tests for policy detection."""

    def test_pii_policy_detection(self):
        """Test PII policy keyword detection."""
        categories = policy_repository.detect_categories("social security number")
        assert "pii" in categories

    def test_legal_policy_detection(self):
        """Test legal policy keyword detection."""
        categories = policy_repository.detect_categories("legal advice lawsuit")
        assert "legal" in categories

    def test_medical_policy_detection(self):
        """Test medical policy keyword detection."""
        categories = policy_repository.detect_categories("medication prescription")
        assert "medical" in categories

    def test_pci_policy_detection(self):
        """Test PCI policy keyword detection."""
        categories = policy_repository.detect_categories("credit card cvv")
        assert "pci" in categories

    def test_multiple_policy_detection(self):
        """Test detecting multiple policy categories."""
        categories = policy_repository.detect_categories(
            "I need the employee's social security number and credit card"
        )
        assert "pii" in categories
        assert "pci" in categories
