from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.embeddings import get_embeddings
from retrieval.vector_store import get_vector_store


EVAL_PATH = PROJECT_ROOT / os.environ.get("CONTEXTGUARD_RETRIEVAL_EVAL_FILE", "evaluation/retrieval_eval_v2.json")
K_VALUES = (1, 3, 5)


def main() -> None:
    with EVAL_PATH.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)

    if vector_store._collection.count() == 0:
        raise RuntimeError("The vector index is empty.")

    aggregate = []

    for item in questions:
        question_id = item["id"]
        query = item["question"]

        gold_keys = {
            (entry["source"], entry["page_number"])
            for entry in item["gold_sources"]
        }

        if not gold_keys:
            raise ValueError(
                f"{question_id} has no verified gold source pages."
            )

        results = vector_store.similarity_search_with_score(
            query,
            k=max(K_VALUES),
        )

        # Match both the source filename and physical page number.
        retrieved_keys = [
            (
                document.metadata.get("source"),
                document.metadata.get("page_number"),
            )
            for document, _ in results
        ]

        first_relevant_rank = next(
            (
                rank
                for rank, key in enumerate(
                    retrieved_keys,
                    start=1,
                )
                if key in gold_keys
            ),
            None,
        )

        reciprocal_rank = (
            1.0 / first_relevant_rank
            if first_relevant_rank is not None
            else 0.0
        )

        metrics = {
            "id": question_id,
            "hit": {},
            "page_recall": {},
            "mrr5": reciprocal_rank,
        }

        print("\n" + "=" * 70)
        print(f"{question_id}: {query}")
        print(f"Gold source-pages: {sorted(gold_keys)}")

        print("Retrieved results:")

        for rank, ((document, distance), key) in enumerate(
            zip(results, retrieved_keys),
            start=1,
        ):
            print(
                f"  Rank {rank} | Source={key[0]} | "
                f"Page={key[1]} | Distance={distance:.4f}"
            )
            print(
                "    "
                + document.page_content[:160].replace("\n", " ")
            )

        for k in K_VALUES:
            top_keys = retrieved_keys[:k]

            # Hit@k: was any expected source-page found?
            hit = int(any(key in gold_keys for key in top_keys))

            # Page Recall@k: how many unique expected source-pages
            # were retrieved within the top k chunks?
            retrieved_unique = set(top_keys)
            recall = (
                len(retrieved_unique.intersection(gold_keys))
                / len(gold_keys)
            )

            metrics["hit"][k] = hit
            metrics["page_recall"][k] = recall

            print(
                f"Hit@{k}: {hit} | "
                f"Source-page Recall@{k}: {recall:.3f}"
            )

        print(f"MRR@5: {reciprocal_rank:.3f}")
        aggregate.append(metrics)

    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print(f"Questions evaluated: {len(aggregate)}")

    for k in K_VALUES:
        print(
            f"Hit@{k}: "
            f"{mean(row['hit'][k] for row in aggregate):.3f}"
        )
        print(
            f"Source-page Recall@{k}: "
            f"{mean(row['page_recall'][k] for row in aggregate):.3f}"
        )

    print(
        "MRR@5: "
        f"{mean(row['mrr5'] for row in aggregate):.3f}"
    )


if __name__ == "__main__":
    main()