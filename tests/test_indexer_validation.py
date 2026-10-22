import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.indexer import MAX_PDF_BYTES, index_uploaded_pdf


def test_rejects_empty_upload_before_indexing():
    with pytest.raises(ValueError, match="uploaded file is empty"):
        index_uploaded_pdf(b"", "empty.pdf", None)


def test_rejects_oversized_upload_before_indexing():
    oversized_content = b"x" * (MAX_PDF_BYTES + 1)

    with pytest.raises(ValueError, match="20 MB"):
        index_uploaded_pdf(oversized_content, "large.pdf", None)


def test_rejects_non_pdf_filename_before_indexing():
    with pytest.raises(ValueError, match="Only PDF files"):
        index_uploaded_pdf(b"some content", "notes.txt", None)


def test_rejects_unreadable_pdf_before_indexing():
    invalid_pdf = b"This is not a valid PDF document."

    with pytest.raises(ValueError, match="not a readable PDF"):
        index_uploaded_pdf(invalid_pdf, "broken.pdf", None)


def test_rejects_password_protected_pdf_before_indexing():
    import pymupdf

    document = pymupdf.open()
    document.new_page()

    try:
        locked_pdf = document.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="test-owner",
            user_pw="test-user",
        )
    finally:
        document.close()

    with pytest.raises(ValueError, match="password-protected"):
        index_uploaded_pdf(locked_pdf, "locked.pdf", None)
