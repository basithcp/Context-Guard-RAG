from __future__ import annotations

import hashlib
from pathlib import Path

import pymupdf
from langchain_core.documents import Document

def calculate_sha256(file_path : path) -> str :
    """Calculate a stable SHA-256 for a document."""
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda : file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()

def load_pdf_pages(pdf_path : str | path) -> list[Document]:
    """
    Extract a PDF into one LangChain Document per physical PDF page.

    Metadata conventions:
    - page_number: one-based PDF viewer page number
    - page_index: zero-based index used by Python
    - source: original PDF filename
    - document_id: SHA-256 of the PDF file
    """

    path = Path(pdf_path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")
    
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, received: {path.name}")

    document_id = calculate_sha256(path)
    page_documents: list[Document] = []

    try:
        with pymupdf.open(path) as pdf:
            if pdf.needs_pass:
                raise ValueError(
                    f"The PDF is password-protected: {path.name}"
                )
            
            total_pages = pdf.page_count

            for page_index, page in enumerate(pdf):
                #Sort text into a more natural reading order where possible
                text = page.get_text("text", sort=True).strip()

                metadata = {
                    "source" : path.name,
                    "document_id" : document_id,
                    "page_number" : page_index + 1,
                    "page_index" : page_index,
                    "total_pages" : total_pages,
                }

                page_documents.append(
                    Document(
                        page_content = text,
                        metadata = metadata,
                    )
                )

    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Could not extract text from PDF: {path.name}"
        ) from exc
    
    if not page_documents:
        raise ValueError(
            f"The PDF contains no pages: {path.name}"
        )
    
    return page_documents


