from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


# --------------------------------------------------
# Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from ingestion.indexer import index_uploaded_pdf

from citations.validator import (  # noqa: E402
    CitationValidationError,
    render_answer,
    validate_answer,
)
from generation.answer_generator import (  # noqa: E402
    ABSTENTION_TEXT,
    build_evidence,
    generate_grounded_answer,
    verify_answer_support,
)
from retrieval.embeddings import get_embeddings  # noqa: E402
from retrieval.vector_store import get_vector_store  # noqa: E402


# --------------------------------------------------
# Streamlit configuration
# --------------------------------------------------

st.set_page_config(
    page_title="ContextGuard",
    page_icon="📚",
    layout="wide",
)

st.title("📚 ContextGuard")
st.caption(
    "Ask questions about your indexed documents and inspect "
    "the source passages behind every answer."
)


with st.expander("How ContextGuard verifies answers"):
    st.markdown(
        """
        1. **Retrieve:** Find relevant passages in ChromaDB.
        2. **Generate:** Ask the LLM to produce claims linked to evidence IDs.
        3. **Validate citations:** Confirm that every cited evidence ID
           belongs to the retrieved passages.
        4. **Audit support:** Use an independent LLM check to assess
           whether the cited passages support the answer.
        5. **Display evidence:** Show the original text, filename,
           and physical PDF page number.

        Citation validation does not prove factual correctness. The
        evidence audit is also model-based and can make mistakes.
        """
    )


# --------------------------------------------------
# Load reusable local resources
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading the embedding model and index...")
def load_resources():
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)
    return vector_store

def get_indexed_document_summaries(vector_store) -> list[dict]:
    """Summarize the documents currently stored in ChromaDB."""

    stored = vector_store.get(include=["metadatas"])
    summaries = {}

    for metadata in stored.get("metadatas") or []:
        if not metadata:
            continue

        document_id = metadata.get("document_id")
        source = metadata.get("source", "Unknown document")
        key = document_id or source

        if key not in summaries:
            summaries[key] = {
                "source": source,
                "chunks": 0,
                "pages": set(),
                "total_pages": metadata.get("total_pages", 0),
            }

        entry = summaries[key]
        entry["chunks"] += 1

        page_number = metadata.get("page_number")
        if isinstance(page_number, int):
            entry["pages"].add(page_number)

    result = []

    for item in summaries.values():
        result.append(
            {
                "source": item["source"],
                "chunks": item["chunks"],
                "pages_with_text": len(item["pages"]),
                "total_pages": item["total_pages"],
            }
        )

    return sorted(
        result,
        key=lambda item: item["source"].lower(),
    )

# --------------------------------------------------
# Load the persistent vector store
# --------------------------------------------------

try:
    vector_store = load_resources()

except Exception as exc:
    print(
        "ContextGuard resource-loading error: "
        f"{type(exc).__name__}: {exc}"
    )

    st.error(
        "ContextGuard could not load its vector index. "
        "Check the terminal output."
    )
    st.stop()


# --------------------------------------------------
# PDF upload and indexing sidebar
# --------------------------------------------------

with st.sidebar:
    st.header("Document Library")

    st.caption(
        "Upload PDFs to add them to the persistent local index."
    )

    st.caption(
        "Maximum 20 MB per PDF and 40 MB total per indexing batch."
    )

    uploaded_files = st.file_uploader(
        "Choose PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        max_upload_size=20,
        key="contextguard_pdf_uploader",
    )

    st.caption(
        "PDF files and embeddings are stored locally. Retrieved "
        "text is sent to the configured Groq API for answer "
        "generation and evidence auditing."
    )

    if st.button(
        "Index Uploaded PDFs",
        disabled=not uploaded_files,
        use_container_width=True,
    ):
        total_upload_bytes = sum(
            uploaded_file.size
            for uploaded_file in uploaded_files
        )

        if total_upload_bytes > 40 * 1024 * 1024:
            st.error(
                "The combined upload exceeds 40 MB. "
                "Please index fewer or smaller PDFs in this batch."
            )

        else:
            for uploaded_file in uploaded_files:
                try:
                    with st.spinner(
                        f"Indexing {uploaded_file.name}..."
                    ):
                        result = index_uploaded_pdf(
                            file_bytes=uploaded_file.getvalue(),
                            original_filename=uploaded_file.name,
                            vector_store=vector_store,
                        )

                    if result["status"] == "already_indexed":
                        st.info(
                            f"{uploaded_file.name}: already indexed. "
                            "No additional chunks were added."
                        )

                    else:
                        st.success(
                            f"{uploaded_file.name}: indexed "
                            f"{result['added']} new chunks."
                        )

                    st.caption(
                        f"Source: {result['source']} | "
                        f"Pages: {result['total_pages']} | "
                        f"Pages with text: {result['text_pages']} | "
                        f"Chunks in document: {result['chunks']}"
                    )

                except Exception as exc:
                    st.error(
                        f"Could not index {uploaded_file.name}: "
                        f"{type(exc).__name__}: {exc}"
                    )


# --------------------------------------------------
# Display the current indexed-document inventory
# --------------------------------------------------

index_count = vector_store._collection.count()

with st.sidebar:
    st.divider()

    st.metric(
        "Indexed chunks",
        index_count,
    )

    indexed_documents = get_indexed_document_summaries(
        vector_store
    )

    with st.expander(
        f"Indexed documents ({len(indexed_documents)})",
        expanded=False,
    ):
        if not indexed_documents:
            st.write("No documents have been indexed yet.")

        for document in indexed_documents:
            st.markdown(f"**{document['source']}**")

            st.caption(
                f"{document['chunks']} chunks · "
                f"{document['pages_with_text']}/"
                f"{document['total_pages']} pages with text"
            )


# Do not stop the app until the upload controls have been rendered.
if index_count == 0:
    st.info(
        "The document index is empty. Upload a PDF using the sidebar "
        "to begin."
    )
    st.stop()

st.caption(f"Indexed chunks available: {index_count}")

# --------------------------------------------------
# Chat state
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# Answer-generation pipeline
# --------------------------------------------------

def answer_question(question: str) -> dict:
    """
    Retrieve evidence, generate an answer, and validate citations.

    Returns cited evidence separately from other retrieved chunks.
    """

    retrieved_documents = vector_store.similarity_search(
        question,
        k=5,
    )

    evidence = build_evidence(retrieved_documents)

    if not evidence:
        return {
            "answer": ABSTENTION_TEXT,
            "evidence": [],
            "cited_evidence": [],
            "other_evidence": [],
            "status": "Abstained: no evidence retrieved.",
            "audit_checks": [],
        }

    payload = generate_grounded_answer(
        question=question,
        evidence=evidence,
    )

    validated = validate_answer(payload, evidence)

    # Abstentions have no cited claims.
    if validated["abstain"]:
        return {
            "answer": ABSTENTION_TEXT,
            "evidence": evidence,
            "cited_evidence": [],
            "other_evidence": evidence,
            "status": (
                "Abstained: the model found insufficient evidence."
            ),
            "audit_checks": [],
        }

    # Verify whether the cited evidence supports the answer.
    audit = verify_answer_support(
        question=question,
        answer=validated,
        evidence=evidence,
    )

    # Suppress the answer if evidence verification fails.
    if not audit["supported"]:
        return {
            "answer": (
                "I cannot find a sufficiently supported answer "
                "in the provided documents."
            ),
            "evidence": evidence,
            "cited_evidence": [],
            "other_evidence": evidence,
            "status": (
                "Evidence-support audit failed or could not be completed. "
                "The proposed answer was suppressed."
            ),
            "audit_checks": audit.get("checks", []),
        }

    # Collect the evidence IDs referenced by validated answer claims.
    cited_ids = {
        source_id
        for claim in validated["claims"]
        for source_id in claim["source_ids"]
    }

    cited_evidence = [
        item
        for item in evidence
        if item["source_id"] in cited_ids
    ]

    other_evidence = [
        item
        for item in evidence
        if item["source_id"] not in cited_ids
    ]

    return {
        "answer": render_answer(validated, evidence),
        "evidence": evidence,
        "cited_evidence": cited_evidence,
        "other_evidence": other_evidence,
        "status": (
            "Citation provenance passed. "
            "Evidence-support audit passed (model-based)."
        ),
        "audit_checks": audit.get("checks", []),
    }

# --------------------------------------------------
# Render previous conversation messages
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":

            st.caption(message.get("status", ""))

            evidence = message.get("evidence", [])

        cited_evidence = message.get("cited_evidence", [])

        other_evidence = message.get("other_evidence")

        # Support messages created before this UI change.
        if other_evidence is None:
            other_evidence = message.get("evidence", [])

        if cited_evidence:
            with st.expander(
                f"Cited evidence ({len(cited_evidence)} chunks)",
                expanded=True,
            ):
                for item in cited_evidence:
                    label = (
                        f"[{item['source_id']}] "
                        f"{item['source']}, "
                        f"page {item['page_number']}"
                    )

                    with st.expander(label):
                        st.text(item["text"])

        elif other_evidence:
            st.caption(
                "No source chunks were cited in the displayed answer."
            )

        if other_evidence:
            with st.expander(
                f"Other retrieved chunks ({len(other_evidence)})",
                expanded=False,
            ):
                for item in other_evidence:
                    label = (
                        f"[{item['source_id']}] "
                        f"{item['source']}, "
                        f"page {item['page_number']}"
                    )

                    with st.expander(label):
                        st.text(item["text"])


# --------------------------------------------------
# User input
# --------------------------------------------------

with st.form("question_form", clear_on_submit=True):
    question = st.text_input(
        "Ask a question about the indexed document",
        placeholder="Example: What does the pwd command display?",
    )

    submitted = st.form_submit_button(
        "Ask ContextGuard",
        type="primary",
    )


if submitted:

    question = question.strip()

    if not question:
        st.warning("Please enter a question.")

    else:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.spinner(
            "Retrieving evidence, generating an answer, "
            "and checking citations..."
        ):
            try:
                result = answer_question(question)

            except CitationValidationError as exc:
                print(f"Citation validation error: {exc}")

                result = {
                    "answer": ABSTENTION_TEXT,
                    "evidence": [],
                    "status": (
                        "Citation validation failed. "
                        "The unvalidated answer was suppressed."
                    ),
                    "audit_checks": [],
                }

            except Exception as exc:
                # Log diagnostics in the terminal, not API secrets
                # or a traceback in the conversation UI.
                print(
                    "ContextGuard answer error: "
                    f"{type(exc).__name__}: {exc}"
                )

                result = {
                    "answer": (
                        "I could not safely complete the answer "
                        "verification. Please try again later."
                    ),
                    "evidence": [],
                    "status": (
                        "Generation or verification failed. "
                        "No unverified answer was displayed."
                    ),
                    "audit_checks": [],
                }

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "evidence": result["evidence"],
                "cited_evidence": result["cited_evidence"],
                "other_evidence": result["other_evidence"],
                "status": result["status"],
                "audit_checks": result["audit_checks"],
            }
        )

        st.rerun()