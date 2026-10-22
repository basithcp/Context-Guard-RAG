from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from citations.validator import (
    CitationValidationError,
    render_answer,
    validate_answer,
)
from generation.answer_generator import (
    ABSTENTION_TEXT,
    build_evidence,
    generate_grounded_answer,
    verify_answer_support,
)
from retrieval.embeddings import get_embeddings
from retrieval.vector_store import get_vector_store


EVAL_PATH = PROJECT_ROOT / os.environ.get("CONTEXTGUARD_EVAL_FILE", "evaluation/answer_eval_v1.json")
TOP_K = 5


def source_page_pairs(documents: list) -> list[tuple[str, int]]:
    """Return source filename and physical page for each chunk."""
    return [
        (
            document.metadata.get("source", "UNKNOWN"),
            document.metadata.get("page_number", -1),
        )
        for document in documents
    ]


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    with EVAL_PATH.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    # Optionally evaluate selected question IDs, e.g. A002 and A008.
    selected_ids = set(sys.argv[1:])

    if selected_ids:
        available_ids = {
            item["id"] for item in questions
        }

        unknown_ids = selected_ids - available_ids

        if unknown_ids:
            raise ValueError(
                f"Unknown evaluation IDs: {sorted(unknown_ids)}"
            )

        questions = [
            item for item in questions
            if item["id"] in selected_ids
        ]

        print("Selected evaluation IDs:", sorted(selected_ids))

    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)

    if vector_store._collection.count() == 0:
        raise RuntimeError("The ChromaDB collection is empty.")

    source_covered = []
    accepted_answerable = []
    unanswerable_safe_outcomes = []
    audit_passes = 0
    audit_failures = 0
    audit_errors = 0

    print("CONTEXTGUARD ANSWER EVALUATION")
    print(f"Indexed chunks: {vector_store._collection.count()}")
    print(f"Evaluation questions: {len(questions)}")

    for item in questions:
        question_id = item["id"]
        question = item["question"]
        answerability = item["answerability"]

        gold_sources = {
            (source["source"], source["page_number"])
            for source in item["gold_sources"]
        }

        print("\n" + "=" * 75)
        print(f"{question_id}: {question}")

        # Stage 1: Retrieve evidence.
        try:
            retrieved_documents = vector_store.similarity_search(
                question,
                k=TOP_K,
            )

            retrieved_pairs = source_page_pairs(
                retrieved_documents
            )

            gold_found = (
                any(pair in gold_sources for pair in retrieved_pairs)
                if gold_sources
                else None
            )

            if gold_found is not None:
                source_covered.append(int(gold_found))

            evidence = build_evidence(retrieved_documents)

            print("Retrieved source-pages:")
            for source, page in retrieved_pairs:
                print(f"  {source}, page {page}")

            if gold_found is not None:
                print(
                    "Gold source found in top five:",
                    gold_found,
                )

            # Stage 2: Generate a structured answer.
            payload = generate_grounded_answer(
                question=question,
                evidence=evidence,
            )
            print("\nGENERATED STRUCTURED PAYLOAD:")
            print(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                )
            )
            # Stage 3: Validate citation IDs and answer structure.
            validated = validate_answer(payload, evidence)

            if not validated["abstain"]:
                evidence_by_id = {
                    item["source_id"]: item
                    for item in evidence
                }

                print("\nCLAIM-TO-EVIDENCE MAPPING")

                for claim_index, claim in enumerate(
                    validated["claims"],
                    start=1,
                ):
                    print(f"\nClaim {claim_index}: {claim['text']}")

                    for source_id in claim["source_ids"]:
                        source = evidence_by_id[source_id]

                        print(
                            f"  [{source_id}] "
                            f"{source['source']}, "
                            f"page {source['page_number']}"
                        )

                        print(
                            "  Evidence: "
                            + source["text"][:300].replace("\n", " ")
                        )
                        
            if validated["abstain"]:
                outcome = "abstained"
                answer_text = ABSTENTION_TEXT

            else:
                # Stage 4: Audit the cited evidence.
                audit = verify_answer_support(
                    question=question,
                    answer=validated,
                    evidence=evidence,
                )

                if audit.get("error"):
                    audit_errors += 1
                    outcome = "verification_error"
                    answer_text = (
                        "Answer suppressed because verification "
                        "could not be completed."
                    )

                    print("Audit error:", audit["error"])

                elif audit["supported"]:
                    audit_passes += 1
                    outcome = "accepted"
                    answer_text = render_answer(
                        validated,
                        evidence,
                    )

                else:
                    audit_failures += 1
                    outcome = "suppressed"
                    answer_text = (
                        "I cannot find a sufficiently supported answer "
                        "in the provided documents."
                    )

                    for check in audit.get("checks", []):
                        if not check["supported"]:
                            print(
                                f"Unsupported claim "
                                f"{check['claim_index']}: "
                                f"{check['reason']}"
                            )

            print("\nPipeline outcome:", outcome)
            print("Displayed answer:")
            print(answer_text)

            print("\nReference answer:")
            print(item["reference_answer"])

            # This is a coverage measure, not an accuracy measure.
            if answerability == "answerable":
                accepted = outcome == "accepted"
                accepted_answerable.append(int(accepted))

                print(
                    "Answerable-question coverage:",
                    "accepted answer" if accepted
                    else "no accepted answer",
                )

            elif answerability == "unanswerable":
                safe = outcome in {
                    "abstained",
                    "suppressed",
                    "verification_error",
                }

                unanswerable_safe_outcomes.append(int(safe))

                print(
                    "Unanswerable-question handling:",
                    "safe" if safe else "UNSAFE: answer accepted",
                )

            elif answerability == "partially_supported":
                print(
                    "Manual review required: check whether the answer "
                    "distinguishes source facts from unsupported inference."
                )

        except (CitationValidationError, ValueError) as exc:
            print("Validation/generation failure:", exc)

        except Exception as exc:
            print(
                f"Evaluation error: {type(exc).__name__}: {exc}"
            )

    print("\n" + "=" * 75)
    print("AGGREGATE RESULTS")

    if source_covered:
        print(
            f"Gold source-page coverage@{TOP_K}: "
            f"{sum(source_covered) / len(source_covered):.3f}"
        )

    if accepted_answerable:
        print(
            "Accepted-answer coverage on answerable questions: "
            f"{sum(accepted_answerable) / len(accepted_answerable):.3f}"
        )

    if unanswerable_safe_outcomes:
        print(
            "Safe handling rate on unanswerable questions: "
            f"{sum(unanswerable_safe_outcomes) / len(unanswerable_safe_outcomes):.3f}"
        )

    print(f"Evidence audits passed: {audit_passes}")
    print(f"Evidence audits rejected claims: {audit_failures}")
    print(f"Evidence audit errors: {audit_errors}")

    print("\nImportant:")
    print(
        "An accepted answer is not automatically a correct answer. "
        "Manually compare outputs with the reference answers and "
        "the cited source text."
    )


if __name__ == "__main__":
    main()