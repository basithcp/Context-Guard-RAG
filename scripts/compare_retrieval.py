from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.chunker import chunk_pages
from ingestion.pdf_loader import load_pdf_pages
from retrieval.embeddings import get_embeddings
from retrieval.hybrid_search import HybridRetriever
from retrieval.vector_store import get_vector_store


PDF_PATH = PROJECT_ROOT / "data" / "raw" / "TLCL-25.12A.pdf"
EVAL_PATH = PROJECT_ROOT / "evaluation" / "retrieval_eval.json"


def page_list(results: list[dict]) -> list[int]:
    return [
        int(result["document"].metadata["page_number"])
        for result in results
    ]


def hit_at_k(
    pages: list[int],
    gold_pages: set[int],
    k: int,
) -> int:
    return int(bool(set(pages[:k]) & gold_pages))


def page_recall_at_k(
    pages: list[int],
    gold_pages: set[int],
    k: int,
) -> float:
    return len(set(pages[:k]) & gold_pages) / len(gold_pages)


def reciprocal_rank_at_k(
    pages: list[int],
    gold_pages: set[int],
    k: int,
) -> float:
    for rank, page in enumerate(pages[:k], start=1):
        if page in gold_pages:
            return 1.0 / rank

    return 0.0


def main() -> None:
    with EVAL_PATH.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    print("Loading PDF and rebuilding the in-memory BM25 index...")
    pages = load_pdf_pages(PDF_PATH)
    chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=50)

    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)

    count = vector_store._collection.count()

    if count != len(chunks):
        raise RuntimeError(
            f"Index contains {count} records but {len(chunks)} "
            "chunks were generated. Resolve this before comparing."
        )

    retriever = HybridRetriever(chunks, vector_store)

    all_metrics = {
        "dense": [],
        "bm25": [],
        "hybrid": [],
    }

    for item in questions:
        query = item["question"]
        gold_pages = set(item["gold_pages"])

        dense = retriever.dense_search(query, k=5)
        sparse = retriever.sparse_search(query, k=5)
        hybrid = retriever.search(query, k=5, candidate_k=20)

        result_sets = {
            "dense": dense,
            "bm25": sparse,
            "hybrid": hybrid,
        }

        print("\n" + "=" * 70)
        print(f"{item['id']}: {query}")
        print(f"Gold pages: {sorted(gold_pages)}")

        for name, results in result_sets.items():
            retrieved_pages = page_list(results)

            metrics = {
                "hit1": hit_at_k(
                    retrieved_pages, gold_pages, 1
                ),
                "hit3": hit_at_k(
                    retrieved_pages, gold_pages, 3
                ),
                "hit5": hit_at_k(
                    retrieved_pages, gold_pages, 5
                ),
                "recall5": page_recall_at_k(
                    retrieved_pages, gold_pages, 5
                ),
                "mrr5": reciprocal_rank_at_k(
                    retrieved_pages, gold_pages, 5
                ),
            }

            all_metrics[name].append(metrics)

            print(f"\n{name.upper()} pages: {retrieved_pages}")
            print(
                f"Hit@1={metrics['hit1']} | "
                f"Hit@3={metrics['hit3']} | "
                f"Hit@5={metrics['hit5']} | "
                f"Page Recall@5={metrics['recall5']:.3f} | "
                f"MRR@5={metrics['mrr5']:.3f}"
            )

        print("\nHybrid top results:")
        for rank, result in enumerate(hybrid, start=1):
            document = result["document"]
            print(
                f"  Rank {rank} | "
                f"Page {document.metadata['page_number']} | "
                f"RRF={result['rrf_score']:.5f} | "
                f"Dense rank={result['dense_rank']} | "
                f"BM25 rank={result['bm25_rank']}"
            )
            print(
                "  "
                + document.page_content[:180].replace("\n", " ")
            )

    print("\n" + "=" * 70)
    print("AGGREGATE COMPARISON")

    for name, rows in all_metrics.items():
        print(f"\n{name.upper()}")

        for metric in (
            "hit1",
            "hit3",
            "hit5",
            "recall5",
            "mrr5",
        ):
            print(
                f"{metric}: "
                f"{mean(row[metric] for row in rows):.3f}"
            )


if __name__ == "__main__":
    main()