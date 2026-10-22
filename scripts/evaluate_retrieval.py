from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.embeddings import get_embeddings
from retrieval.vector_store import get_vector_store


EVAL_PATH = PROJECT_ROOT / "evaluation" / "retrieval_eval.json"
K_VALUES = (1, 3, 5)


def main() -> None:
    if not EVAL_PATH.is_file():
        raise FileNotFoundError(f"Evaluation file not found: {EVAL_PATH}")

    with EVAL_PATH.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if not questions:
        raise ValueError("The evaluation dataset is empty.")

    print("Loading embedding model...")
    embeddings = get_embeddings()

    print("Opening persistent ChromaDB collection...")
    vector_store = get_vector_store(embeddings)

    collection_count = vector_store._collection.count()

    if collection_count == 0:
        raise RuntimeError(
            "The ChromaDB collection is empty. Build the index first."
        )

    all_metrics = []

    for item in questions:
        question_id = item["id"]
        query = item["question"]
        gold_pages = set(item["gold_pages"])

        if not gold_pages:
            raise ValueError(
                f"{question_id} has no gold source pages."
            )

        results = vector_store.similarity_search_with_score(
            query,
            k=max(K_VALUES),
        )

        retrieved_pages = [
            int(document.metadata["page_number"])
            for document, _ in results
        ]

        first_relevant_rank = next(
            (
                rank
                for rank, page in enumerate(retrieved_pages, start=1)
                if page in gold_pages
            ),
            None,
        )

        reciprocal_rank = (
            1.0 / first_relevant_rank
            if first_relevant_rank is not None
            else 0.0
        )

        query_metrics = {
            "id": question_id,
            "hit": {},
            "page_recall": {},
            "rr_at_5": reciprocal_rank,
        }

        print("\n" + "=" * 65)
        print(f"{question_id}: {query}")
        print(f"Gold pages: {sorted(gold_pages)}")
        print(f"Retrieved pages: {retrieved_pages}")

        for k in K_VALUES:
            top_pages = set(retrieved_pages[:k])

            hits = len(top_pages.intersection(gold_pages))
            page_recall = hits / len(gold_pages)
            hit = int(hits > 0)

            query_metrics["hit"][k] = hit
            query_metrics["page_recall"][k] = page_recall

            print(
                f"Hit@{k}: {hit} | "
                f"Page Recall@{k}: {page_recall:.3f}"
            )

        print(f"Reciprocal Rank@5: {reciprocal_rank:.3f}")

        # Inspect the evidence behind the top retrieved results.
        for rank, (document, distance) in enumerate(results, start=1):
            print(
                f"\n  Rank {rank} | "
                f"Page {document.metadata['page_number']} | "
                f"Cosine distance {distance:.4f}"
            )
            print(
                "  "
                + document.page_content[:180].replace("\n", " ")
            )

        all_metrics.append(query_metrics)

    print("\n" + "=" * 65)
    print("AGGREGATE RETRIEVAL RESULTS")
    print(f"Evaluation questions: {len(all_metrics)}")

    for k in K_VALUES:
        avg_hit = mean(
            item["hit"][k] for item in all_metrics
        )
        avg_recall = mean(
            item["page_recall"][k] for item in all_metrics
        )

        print(
            f"Hit@{k}: {avg_hit:.3f} | "
            f"Mean Page Recall@{k}: {avg_recall:.3f}"
        )

    mrr = mean(item["rr_at_5"] for item in all_metrics)
    print(f"MRR@5: {mrr:.3f}")

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
