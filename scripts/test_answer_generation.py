from __future__ import annotations

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
    build_evidence,
    generate_grounded_answer,
    verify_answer_support,
)
from retrieval.embeddings import get_embeddings
from retrieval.vector_store import get_vector_store


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)

    if vector_store._collection.count() == 0:
        raise RuntimeError(
            "The ChromaDB collection is empty. Build the index first."
        )

    test_questions = [
        "What does the pwd command display?",
        "How can command output be redirected to a file?",
        "Which environment variable represents the current working directory?",
        "What is the current IP address assigned to my Kali machine?",
    ]

    for question in test_questions:
        print("\n" + "=" * 75)
        print("QUESTION:", question)

        # Retrieve using the current dense-search baseline.
        retrieved_documents = vector_store.similarity_search(
            question,
            k=5,
        )

        evidence = build_evidence(retrieved_documents)

        print("\nRETRIEVED EVIDENCE")

        for item in evidence:
            print(
                f"\n[{item['source_id']}] "
                f"{item['source']}, "
                f"page {item['page_number']}"
            )
            print(item["text"][:250].replace("\n", " "))

        print("\nGENERATING ANSWER...")

        try:
            payload = generate_grounded_answer(
                question=question,
                evidence=evidence,
            )

            # First validate the output structure and source IDs.
            validated = validate_answer(payload, evidence)

            if validated["abstain"]:
                print("\nANSWER")
                print(render_answer(validated, evidence))
                print(
                    "\nOUTPUT VALIDATION: PASS "
                    "(valid abstention; no factual claims)"
                )
                continue

            # Next independently audit the cited evidence.
            audit = verify_answer_support(
                question=question,
                answer=validated,
                evidence=evidence,
            )

            if audit["supported"]:
                print("\nANSWER")
                print(render_answer(validated, evidence))
                print("\nCITATION PROVENANCE: PASS")
                print("EVIDENCE SUPPORT AUDIT: PASS")
            else:
                print("\nANSWER")
                print(
                    "I cannot find a sufficiently supported answer "
                    "in the provided documents."
                )
                print("\nCITATION PROVENANCE: PASS")
                print("EVIDENCE SUPPORT AUDIT: FAIL")
                print(
                    "The proposed answer was suppressed because "
                    "its evidence could not be verified."
                )

                if audit.get("error"):
                    print("Audit detail:", audit["error"])

                for check in audit.get("checks", []):
                    if not check["supported"]:
                        print(
                            f"Unsupported claim {check['claim_index']}: "
                            f"{check['reason']}"
                        )

        except (CitationValidationError, ValueError) as exc:
            print("\nVALIDATION FAILED")
            print(exc)
            print(
                "Fail closed: do not display the model's "
                "unvalidated claims as a verified answer."
            )


if __name__ == "__main__":
    main()