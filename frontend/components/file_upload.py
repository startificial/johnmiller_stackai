"""
File upload components with progress indication.
"""

from typing import Optional

import streamlit as st

from api.client import APIClient, APIError
from api.models import BatchIngestionResponse


def render_file_uploader() -> list:
    """
    Render file upload widget for PDFs.

    Returns:
        List of uploaded file objects
    """
    return st.file_uploader(
        "Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to add to the knowledge base",
    )


def upload_files_with_progress(
    files: list,
    api_client: APIClient,
) -> Optional[BatchIngestionResponse]:
    """
    Upload files and show progress.

    Args:
        files: List of Streamlit UploadedFile objects
        api_client: APIClient instance

    Returns:
        BatchIngestionResponse if successful, None otherwise
    """
    if not files:
        return None

    progress_bar = st.progress(0, text="Preparing upload...")
    status_container = st.empty()

    try:
        # Prepare files for upload
        progress_bar.progress(0.1, text="Uploading files...")

        file_data = [(f.name, f) for f in files]
        result = api_client.ingest_documents(file_data)

        progress_bar.progress(1.0, text="Processing complete!")

        # Display results
        _render_upload_results(result, status_container)

        return result

    except APIError as e:
        progress_bar.empty()
        status_container.error(f"Upload failed: {e.message}")
        return None
    except Exception as e:
        progress_bar.empty()
        status_container.error(f"Unexpected error: {str(e)}")
        return None


def _render_upload_results(result: BatchIngestionResponse, container):
    """
    Render upload results.

    Args:
        result: BatchIngestionResponse
        container: Streamlit container for status message
    """
    # Show summary status
    if result.successful > 0:
        container.success(
            f"Successfully processed {result.successful}/{result.total_files} files"
        )
    elif result.failed > 0:
        container.error(f"All {result.failed} files failed to process")

    # Show errors
    for error in result.errors:
        st.error(f"**{error.filename}:** {error.error}")

    # Show successful documents
    if result.documents:
        st.markdown("**Processed Documents:**")
        for doc in result.documents:
            with st.expander(f"{doc.filename}", expanded=False):
                col1, col2, col3 = st.columns(3)
                col1.metric("Pages", doc.page_count)
                col2.metric("Chunks", doc.chunks_created)
                col3.metric("Embedded", doc.chunks_embedded)
                st.caption(f"Document ID: `{doc.document_id}`")


def render_upload_section(api_client: APIClient) -> Optional[BatchIngestionResponse]:
    """
    Render complete upload section with button.

    Args:
        api_client: APIClient instance

    Returns:
        BatchIngestionResponse if upload completed, None otherwise
    """
    files = render_file_uploader()

    if files:
        st.caption(f"{len(files)} file(s) selected")
        if st.button("Process Files", type="primary", use_container_width=True):
            return upload_files_with_progress(files, api_client)

    return None
