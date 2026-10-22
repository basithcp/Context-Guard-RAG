from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

# Make imports work when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.chunker import chunk_pages
from ingestion.pdf_loader import load_pdf_pages


PDF_PATH = PROJECT_ROOT / "data" / "raw" / "TLCL-25.12A.pdf"


def main() -> None:
    print("=" * 60)
    print("CONTEXTGUARD: PDF INGESTION TEST")
    print("=" * 60)

    if not PDF_PATH.is_file():
        raise FileNotFoundError(
            f"Expected PDF at: {PDF_PATH}\n"
            "Check the filename and data/raw directory."
        )

    # Stage 1: Extract each physical PDF page.
    pages = load_pdf_pages(PDF_PATH)

    empty_pages = sum(
        not page.page_content.strip()
        for page in pages
    )

    populated_pages = len(pages) - empty_pages

    print("\n[1] PDF EXTRACTION")
    print(f"PDF filename: {PDF_PATH.name}")
    print(f"Physical pages: {len(pages)}")
    print(f"Pages with extractable text: {populated_pages}")
    print(f"Pages with no extractable text: {empty_pages}")
    print(f"Document ID: {pages[0].metadata['document_id']}")

    # Stage 2: Split pages into chunks.
    chunks = chunk_pages(
        pages,
        chunk_size=500,
        chunk_overlap=50,
    )

    print("\n[2] CHUNKING")
    print("Chunk size: 500 characters")
    print("Chunk overlap: 50 characters")
    print(f"Total chunks: {len(chunks)}")

    page_counts = Counter(
        chunk.metadata["page_number"]
        for chunk in chunks
    )

    print(f"Pages represented in chunks: {len(page_counts)}")

    if chunks:
        max_chunk_length = max(
            chunk.metadata["char_count"]
            for chunk in chunks
        )
        print(f"Largest chunk: {max_chunk_length} characters")

    # Stage 3: Display representative output.
    if chunks:
        print("\n[3] FIRST CHUNK")
        first_chunk = chunks[0]

        print("Source:", first_chunk.metadata["source"])
        print("PDF page:", first_chunk.metadata["page_number"])
        print("Chunk index:", first_chunk.metadata["chunk_index"])
        print("Character offset:", first_chunk.metadata["start_index"])
        print("Chunk ID:", first_chunk.metadata["chunk_id"])
        print("\nText preview:")
        print(first_chunk.page_content[:500])
    else:
        raise RuntimeError(
            "No chunks were generated. Inspect the PDF extraction output."
        )

    # Stage 4: Basic metadata checks.
    print("\n[4] METADATA VALIDATION")

    assert len(pages) == pages[0].metadata["total_pages"]
    assert all(
        1 <= page.metadata["page_number"] <= len(pages)
        for page in pages
    )
    assert all(
        chunk.metadata["source"] == PDF_PATH.name
        for chunk in chunks
    )
    assert all(
        chunk.metadata["page_number"] >= 1
        for chunk in chunks
    )
    assert all(
        chunk.page_content.strip()
        for chunk in chunks
    )

    print("PASS: page numbering is within the PDF's physical range.")
    print("PASS: source metadata is preserved.")
    print("PASS: no empty chunks were generated.")
    print("\nIngestion smoke test completed.")


if __name__ == "__main__":
    main()