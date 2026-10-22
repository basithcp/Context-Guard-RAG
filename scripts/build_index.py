from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.chunker import chunk_pages
from ingestion.pdf_loader import load_pdf_pages
from retrieval.embeddings import get_embeddings
from retrieval.vector_store import get_vector_store


PDF_PATH = PROJECT_ROOT / "data" / "raw" / "TLCL-25.12A.pdf"
BATCH_SIZE = 64


def main() -> None:
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    print("Loading PDF pages...")
    pages = load_pdf_pages(PDF_PATH)

    print("Chunking pages...")
    chunks = chunk_pages(
        pages,
        chunk_size=500,
        chunk_overlap=50,
    )

    print(f"Extracted pages: {len(pages)}")
    print(f"Chunks to index: {len(chunks)}")

    if not chunks:
        raise RuntimeError("No chunks were produced.")

    print("\nLoading local embedding model...")
    embeddings = get_embeddings()

    # Test the model before indexing the complete document.
    test_vector = embeddings.embed_query(
        "What does the pwd command display?"
    )

    print(f"Embedding dimensions: {len(test_vector)}")

    if len(test_vector) != 384:
        raise RuntimeError(
            f"Expected 384 dimensions, got {len(test_vector)}"
        )

    print("\nOpening persistent ChromaDB collection...")
    vector_store = get_vector_store(embeddings)

    # Reuse existing chunk IDs so an interrupted run can resume.
    existing_data = vector_store.get(include=["metadatas"])
    existing_ids = set(existing_data["ids"])

    pending_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata["chunk_id"] not in existing_ids
    ]

    print(f"Already stored: {len(existing_ids)}")
    print(f"Remaining to index: {len(pending_chunks)}")

    for start in range(0, len(pending_chunks), BATCH_SIZE):
        batch = pending_chunks[start : start + BATCH_SIZE]

        vector_store.add_documents(
            documents=batch,
            ids=[
                chunk.metadata["chunk_id"]
                for chunk in batch
            ],
        )

        completed = min(
            start + BATCH_SIZE,
            len(pending_chunks),
        )

        print(
            f"Indexed {completed}/{len(pending_chunks)} "
            "new chunks"
        )

    # Reopen the persistent collection to verify storage.
    reopened_store = get_vector_store(embeddings)
    saved_data = reopened_store.get(include=["metadatas"])
    saved_count = len(saved_data["ids"])

    print("\nINDEX VERIFICATION")
    print(f"Expected chunks: {len(chunks)}")
    print(f"Stored chunk records: {saved_count}")

    if saved_count != len(chunks):
        raise RuntimeError(
            "The stored record count differs from the expected count. "
            "Inspect the collection before proceeding."
        )

    print("PASS: all expected chunk records are present.")

    # Retrieval smoke tests: these are diagnostic questions,
    # not our final evaluation benchmark.
    queries = [
        "What does the pwd command display?",
        "How can command output be redirected to a file?",
        "What is the difference between an absolute and relative pathname?",
    ]

    print("\nRETRIEVAL SMOKE TESTS")

    for query in queries:
        print(f"\nQuestion: {query}")

        results = reopened_store.similarity_search_with_score(
            query,
            k=3,
        )

        for rank, (document, distance) in enumerate(
            results,
            start=1,
        ):
            metadata = document.metadata

            print(
                f"\n  Result {rank}"
                f" | PDF page: {metadata['page_number']}"
                f" | cosine distance: {distance:.4f}"
            )
            print(
                "  Text: "
                + document.page_content[:260].replace("\n", " ")
            )

    print("\nStep 4B completed successfully.")


if __name__ == "__main__":
    main()