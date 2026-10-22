from langchain_huggingface import HuggingFaceEmbeddings

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load the local embedding model on the CPU.

    Batch size is deliberately modest for the available RAM.
    """
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device":"cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 10,
        },
        query_encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 1,
        },
        show_progress=True,
    )