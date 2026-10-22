from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.embeddings import get_embeddings
from retrieval.vector_store import get_vector_store


QUERIES = [
    "What is AVISPA?",
    "Which environment variable represents the current working directory?",
]

REPEATS = 5
TOP_K = 5


def main() -> None:
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)

    signatures_by_query = {}

    print("RETRIEVAL REPEATABILITY TEST")

    for query in QUERIES:
        print("\n" + "=" * 70)
        print("Query:", query)

        signatures = []

        for run in range(1, REPEATS + 1):
            results = vector_store.similarity_search_with_score(
                query,
                k=TOP_K,
            )

            signature = tuple(
                (
                    document.metadata.get("source"),
                    document.metadata.get("page_number"),
                    document.metadata.get("chunk_id"),
                )
                for document, _ in results
            )

            signatures.append(signature)

            print(f"\nRun {run}:")

            for rank, (document, distance) in enumerate(
                results,
                start=1,
            ):
                metadata = document.metadata

                print(
                    f"  Rank {rank} | "
                    f"{metadata.get('source')} | "
                    f"page {metadata.get('page_number')} | "
                    f"distance {distance:.4f} | "
                    f"chunk {metadata.get('chunk_id', '?')[:12]}"
                )

        distinct_results = len(set(signatures))
        signatures_by_query[query] = distinct_results

        print(
            f"\nDistinct top-{TOP_K} result lists: "
            f"{distinct_results} across {REPEATS} runs"
        )

    print("\nSUMMARY")

    for query, count in signatures_by_query.items():
        print(f"{query}: {count} distinct result lists")


if __name__ == "__main__":
    main()