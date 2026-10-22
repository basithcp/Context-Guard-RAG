from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pymupdf
from langchain_chroma import Chroma

from ingestion.chunker import chunk_pages
from ingestion.pdf_loader import load_pdf_pages


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = PROJECT_ROOT / "data" / "raw" / "uploads"

MAX_PDF_BYTES = 20 * 1024 * 1024
INDEX_BATCH_SIZE = 32

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def is_internal_filename(
    filename: str,
    document_id: str,
) -> bool:
    """Check whether a source name has our internal hash suffix."""
    stem = Path(filename).stem

    return (
        stem.endswith(f"__{document_id[:8]}")
        or stem.endswith(f"__{document_id[:16]}")
    )


def index_uploaded_pdf(
    file_bytes: bytes,
    original_filename: str,
    vector_store: Chroma,
) -> dict:
    """
    Validate and incrementally index an uploaded PDF.

    - Stores the physical file with a collision-resistant name.
    - Uses the human-readable filename for source citations.
    - Adds only missing chunks.
    - Repairs old hash-suffixed source metadata.
    - Verifies that all expected chunk IDs exist afterward.
    """
    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    if len(file_bytes) > MAX_PDF_BYTES:
        raise ValueError("The maximum allowed PDF size is 20 MB.")

    # Handle both Unix-style and Windows-style path separators.
    supplied_name = Path(
        original_filename.replace("\\", "/")
    ).name

    if Path(supplied_name).suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    document_id = hashlib.sha256(file_bytes).hexdigest()

    # Validate PDF contents, not just the filename extension.
    try:
        with pymupdf.open(
            stream=file_bytes,
            filetype="pdf",
        ) as pdf:
            if pdf.needs_pass:
                raise ValueError(
                    f"{supplied_name} is password-protected."
                )

            total_pages = pdf.page_count

            if total_pages < 1:
                raise ValueError("The PDF contains no pages.")

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"{supplied_name} is not a readable PDF."
        ) from exc

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Create a safe storage filename. This name is not used for citations.
    safe_stem = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        Path(supplied_name).stem,
    ).strip("._-")

    safe_stem = (safe_stem or "document")[:80]

    saved_path = (
        UPLOAD_DIR / f"{safe_stem}__{document_id[:8]}.pdf"
    )

    # Handle a shortened-hash filename collision without overwriting files.
    if saved_path.exists():
        stored_hash = hashlib.sha256(
            saved_path.read_bytes()
        ).hexdigest()

        if stored_hash != document_id:
            saved_path = (
                UPLOAD_DIR / f"{safe_stem}__{document_id[:16]}.pdf"
            )

    if saved_path.exists():
        stored_hash = hashlib.sha256(
            saved_path.read_bytes()
        ).hexdigest()

        if stored_hash != document_id:
            raise RuntimeError(
                "A storage filename collision occurred. "
                "No existing file was overwritten."
            )
    else:
        saved_path.write_bytes(file_bytes)

    # IMPORTANT: load both IDs and metadata before accessing either.
    existing_data = vector_store.get(
        where={"document_id": document_id},
        include=["metadatas"],
    )

    existing_ids = list(existing_data["ids"])
    existing_metadatas = (
        existing_data.get("metadatas") or []
    )

    # These two collections are aligned by position.
    existing_metadata_by_id = {
        chunk_id: metadata
        for chunk_id, metadata in zip(
            existing_ids,
            existing_metadatas,
        )
        if metadata is not None
    }

    existing_sources = [
        metadata.get("source")
        for metadata in existing_metadatas
        if metadata and metadata.get("source")
    ]

    previous_source = (
        existing_sources[0]
        if existing_sources
        else None
    )

    # Repair filenames created by the earlier indexer.
    if previous_source and not is_internal_filename(
        previous_source,
        document_id,
    ):
        # Preserve an established human-readable name.
        source_name = previous_source
    else:
        # Use the filename supplied by the user for new or legacy records.
        source_name = supplied_name

    # Extract one Document per physical PDF page.
    pages = load_pdf_pages(saved_path)

    for page in pages:
        page.metadata["source"] = source_name

    text_pages = sum(
        bool(page.page_content.strip())
        for page in pages
    )

    chunks = chunk_pages(
        pages,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    if not chunks:
        raise ValueError(
            "No extractable text was found in this PDF. "
            "Scanned or image-only PDFs require OCR, which is not "
            "implemented yet."
        )

    chunk_ids = [
        chunk.metadata["chunk_id"]
        for chunk in chunks
    ]

    existing_id_set = set(existing_ids)

    pending_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata["chunk_id"] not in existing_id_set
    ]

    # First, repair old source labels using metadata-only updates.
    ids_to_relabel = []
    metadatas_to_relabel = []

    for chunk in chunks:
        chunk_id = chunk.metadata["chunk_id"]
        old_metadata = existing_metadata_by_id.get(chunk_id)

        if (
            old_metadata is not None
            and old_metadata.get("source") != source_name
        ):
            updated_metadata = dict(old_metadata)
            updated_metadata["source"] = source_name

            ids_to_relabel.append(chunk_id)
            metadatas_to_relabel.append(updated_metadata)

    for start in range(
        0,
        len(ids_to_relabel),
        INDEX_BATCH_SIZE,
    ):
        end = start + INDEX_BATCH_SIZE

        vector_store._collection.update(
            ids=ids_to_relabel[start:end],
            metadatas=metadatas_to_relabel[start:end],
        )

    # Add missing chunks without duplicating previously indexed records.
    for start in range(
        0,
        len(pending_chunks),
        INDEX_BATCH_SIZE,
    ):
        batch = pending_chunks[
            start : start + INDEX_BATCH_SIZE
        ]

        vector_store.add_documents(
            documents=batch,
            ids=[
                chunk.metadata["chunk_id"]
                for chunk in batch
            ],
        )

    # Verify persistence for this document.
    verified_data = vector_store.get(
        where={"document_id": document_id},
        include=["metadatas"],
    )

    saved_ids = set(verified_data["ids"])
    missing_ids = set(chunk_ids) - saved_ids

    if missing_ids:
        raise RuntimeError(
            f"Index verification failed: {len(missing_ids)} "
            "expected chunks are missing."
        )

    status = (
        "indexed"
        if pending_chunks
        else "already_indexed"
    )

    return {
        "status": status,
        "source": source_name,
        "document_id": document_id,
        "total_pages": total_pages,
        "text_pages": text_pages,
        "empty_pages": total_pages - text_pages,
        "chunks": len(chunks),
        "added": len(pending_chunks),
        "relabeled": len(ids_to_relabel),
    }