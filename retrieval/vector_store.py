from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "contextguard_tlcl_minilm_cs500_ov50"


def get_vector_store(embedding_function : Embeddings) -> Chroma:
    """
    Open or create the persistent ContextGuard collection.
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )

