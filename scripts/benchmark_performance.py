from __future__ import annotations

import gc
import resource
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.chunker import chunk_pages
from ingestion.pdf_loader import load_pdf_pages
from retrieval.embeddings import get_embeddings
from retrieval.vector_store import CHROMA_DIR, get_vector_store


PDF_PATH = PROJECT_ROOT / "data" / "raw" / "TLCL-25.12A.pdf"

QUERIES = [
    "What does the pwd command display?",
    "Which environment variable represents the current working directory?",
    "How can command output be redirected to a file?",
    "What is AVISPA?",
]

RETRIEVAL_REPEATS = 7


def directory_size(path: Path) -> int:
    """Return the total size in bytes of files under a directory."""
    if not path.exists():
        return 0

    return sum(
        file.stat().st_size
        for file in path.rglob("*")
        if file.is_file()
    )


def available_ram_gib() -> float | None:
    """Read system-wide available RAM from Linux /proc/meminfo."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as file:
            for line in file:
                if line.startswith("MemAvailable:"):
                    available_kib = int(line.split()[1])
                    return available_kib / (1024 * 1024)
    except (OSError, ValueError):
        return None

    return None


def peak_process_ram_mib() -> float:
    """Return this process's historical peak RSS in MiB on Linux."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024


def percentile95(values: list[float]) -> float:
    """Return the nearest-rank 95th percentile."""
    ordered = sorted(values)
    index = max(0, int(0.95 * len(ordered) + 0.999999) - 1)
    return ordered[index]


def main() -> None:
    print("=" * 70)
    print("CONTEXTGUARD PERFORMANCE BENCHMARK")
    print("=" * 70)

    print(f"Project directory: {PROJECT_ROOT}")
    print(f"Initial available RAM: {available_ram_gib()} GiB")

    disk_before = directory_size(CHROMA_DIR)
    free_before = __import__("shutil").disk_usage(
        PROJECT_ROOT
    ).free

    # 1. Measure PDF extraction.
    print("\n[1] PDF EXTRACTION")

    start = time.perf_counter()
    pages = load_pdf_pages(PDF_PATH)
    extraction_seconds = time.perf_counter() - start

    print(f"Physical pages: {len(pages)}")
    print(
        f"Extraction time: {extraction_seconds:.3f} seconds"
    )

    # 2. Measure text chunking.
    print("\n[2] CHUNKING")

    start = time.perf_counter()
    chunks = chunk_pages(
        pages,
        chunk_size=500,
        chunk_overlap=50,
    )
    chunking_seconds = time.perf_counter() - start

    print(f"Generated chunks: {len(chunks)}")
    print(f"Chunking time: {chunking_seconds:.3f} seconds")

    # 3. Load local embeddings and inspect the existing index.
    print("\n[3] EMBEDDING MODEL AND INDEX")

    start = time.perf_counter()
    embeddings = get_embeddings()
    embedding_model_load_seconds = time.perf_counter() - start

    print(
        "Embedding model initialization: "
        f"{embedding_model_load_seconds:.3f} seconds"
    )

    vector_store = get_vector_store(embeddings)

    stored = vector_store.get(
        include=["documents", "metadatas"]
    )

    indexed_texts = [
        text
        for text in stored["documents"]
        if isinstance(text, str) and text.strip()
    ]

    print(f"Indexed text chunks: {len(indexed_texts)}")
    print(
        "Persistent ChromaDB records: "
        f"{vector_store._collection.count()}"
    )

    if not indexed_texts:
        raise RuntimeError(
            "No stored chunk text is available to benchmark."
        )

    # This is a dry-run embedding benchmark: no vectors are added
    # to the persistent index.
    start = time.perf_counter()
    vectors = embeddings.embed_documents(indexed_texts)
    embedding_seconds = time.perf_counter() - start

    dimensions = len(vectors[0]) if vectors else 0

    print(
        f"Dry-run embeddings generated: {len(vectors)}"
    )
    print(f"Embedding dimensions: {dimensions}")
    print(
        "Embedding generation time: "
        f"{embedding_seconds:.3f} seconds"
    )

    if len(vectors) != len(indexed_texts):
        raise RuntimeError(
            "The generated embedding count does not match "
            "the number of stored chunks."
        )

    if dimensions != 384:
        raise RuntimeError(
            f"Expected 384-dimensional embeddings, got {dimensions}."
        )

    # Release the vectors before measuring retrieval.
    del vectors
    gc.collect()

    # 4. Measure retrieval latency for representative questions.
    print("\n[4] RETRIEVAL LATENCY")

    for query in QUERIES:
        # Warm up the model before recording timings.
        vector_store.similarity_search(query, k=5)

        measurements = []

        for _ in range(RETRIEVAL_REPEATS):
            start = time.perf_counter()

            results = vector_store.similarity_search(
                query,
                k=5,
            )

            elapsed = time.perf_counter() - start
            measurements.append(elapsed * 1000)

        print(f"\nQuery: {query}")
        print(f"Results returned: {len(results)}")
        print(
            f"Mean: {statistics.mean(measurements):.2f} ms"
        )
        print(
            f"Median: {statistics.median(measurements):.2f} ms"
        )
        print(
            f"P95: {percentile95(measurements):.2f} ms"
        )

        if results:
            first = results[0]

            print(
                "Top result: "
                f"{first.metadata.get('source')}, "
                f"page {first.metadata.get('page_number')}"
            )

    # 5. Report RAM and storage measurements.
    print("\n[5] RESOURCE USAGE")

    disk_after = directory_size(CHROMA_DIR)
    free_after = __import__("shutil").disk_usage(
        PROJECT_ROOT
    ).free

    print(f"Peak process RAM: {peak_process_ram_mib():.1f} MiB")
    print(f"Available system RAM: {available_ram_gib()} GiB")
    print(f"ChromaDB directory size: {disk_after / (1024**2):.2f} MiB")
    print(
        "Change in ChromaDB directory size during benchmark: "
        f"{(disk_after - disk_before) / (1024**2):.3f} MiB"
    )
    print(
        "Change in filesystem free space: "
        f"{(free_after - free_before) / (1024**2):.3f} MiB"
    )

    print("\n[6] SUMMARY")

    print(
        "Local extraction + chunking: "
        f"{extraction_seconds + chunking_seconds:.3f} seconds"
    )
    print(
        "Local model initialization + dry-run embedding: "
        f"{embedding_model_load_seconds + embedding_seconds:.3f} seconds"
    )

    print(
        "\nBenchmark completed. No ChromaDB records were intentionally "
        "added, updated, or deleted."
    )


if __name__ == "__main__":
    main()