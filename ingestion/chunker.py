from __future__ import annotations

import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(page_documents: list[Document], chunk_size: int = 500,
chunk_overlap: int = 50,) -> list[Document]:
    """
    Split individual PDF pages into overlapping text chunks.

    Every resulting chunk retains its original document and page metadata.
    Chunks never combine text from different PDF pages.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be greater than zero")
    
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overalp must be non-negative")
    
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap must be smaller than chunk_size")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        strip_whitespace=True,
    )

    all_chunks : list[Document] = []

    for page_document in page_documents:
        # Keep empty pages in the extraction output, but do not embed them.
        if not page_document.page_content.strip():
            continue

        page_chunks = splitter.split_documents([page_document])

        for page_chunk_index, chunk in enumerate(page_chunks):
            metadata = dict(chunk.metadata)

            # This is the chunk's position in the extracted page text.
            start_index = metadata.get("start_index", 0)

            metadata["chunk_index"] = len(all_chunks)
            metadata["page_chunk_index"] = page_chunk_index
            metadata["char_count"] = len(chunk.page_content)

            # Stable for the same document, page, position, and chunk text.

            identity = (
                f"{metadata['document_id']}:"
                f"{metadata['page_index']}:"
                f"{start_index}:"
                f"{chunk.page_content}"
            )

            metadata["chunk_id"] = hashlib.sha256(
                identity.encode('utf-8')
            ).hexdigest()

            all_chunks.append(
                Document(
                    page_content=chunk.page_content,
                    metadata=metadata
                )
            )
    
    return all_chunks