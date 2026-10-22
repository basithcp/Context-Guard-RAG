from __future__ import annotations

import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from citations.validator import render_answer, validate_answer
from generation.answer_generator import (
    ABSTENTION_TEXT,
    build_evidence,
    generate_grounded_answer,
    verify_answer_support,
)
from retrieval.embeddings import get_embeddings
from retrieval.vector_store import get_vector_store


TOP_K = 5


def main() -> None:
    question = (
        " ".join(sys.argv[1:]).strip()
        or "What does the pwd command display?"
    )

    print("CONTEXTGUARD ANSWER-LATENCY BENCHMARK")
    print(f"Question: {question}")

    # Resource loading is measured separately because Streamlit caches it.
    resource_start = time.perf_counter()
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)
    resource_seconds = time.perf_counter() - resource_start

    print(f"Indexed chunks: {vector_store._collection.count()}")
    print(f"Resource initialization: {resource_seconds:.2f} seconds")

    timings = {
        "retrieval": 0.0,
        "evidence preparation": 0.0,
        "answer generation": 0.0,
        "citation validation": 0.0,
        "support audit": 0.0,
    }

    request_start = time.perf_counter()

    # Stage 1: Retrieve relevant document chunks.
    start = time.perf_counter()
    documents = vector_store.similarity_search(question, k=TOP_K)
    timings["retrieval"] = time.perf_counter() - start

    # Stage 2: Convert retrieved chunks into source-labelled evidence.
    start = time.perf_counter()
    evidence = build_evidence(documents)
    timings["evidence preparation"] = time.perf_counter() - start

    if not evidence:
        answer = ABSTENTION_TEXT
        outcome = "Abstained: no evidence retrieved."

    else:
        # Stage 3: Generate structured claims using the configured Groq model.
        start = time.perf_counter()
        payload = generate_grounded_answer(
            question=question,
            evidence=evidence,
        )
        timings["answer generation"] = time.perf_counter() - start

        # Stage 4: Validate that cited source IDs exist in the retrieved evidence.
        start = time.perf_counter()
        validated = validate_answer(payload, evidence)
        timings["citation validation"] = time.perf_counter() - start

        if validated["abstain"]:
            answer = ABSTENTION_TEXT
            outcome = "Abstained: insufficient evidence."

        else:
            # Stage 5: Independently audit whether evidence supports each claim.
            start = time.perf_counter()
            audit = verify_answer_support(
                question=question,
                answer=validated,
                evidence=evidence,
            )
            timings["support audit"] = time.perf_counter() - start

            if not audit["supported"]:
                answer = (
                    "I cannot find a sufficiently supported answer "
                    "in the provided documents."
                )
                outcome = "Answer suppressed: support audit failed."
            else:
                answer = render_answer(validated, evidence)
                outcome = "Accepted: citation and support checks passed."

    request_seconds = time.perf_counter() - request_start

    print("\nRESULT")
    print(f"Outcome: {outcome}")
    print(f"Answer: {answer}")

    print("\nSTAGE TIMINGS")
    for stage, seconds in timings.items():
        print(f"{stage}: {seconds:.3f} seconds")

    print(f"\nTotal Q&A latency: {request_seconds:.3f} seconds")
    print("Note: resource initialization is excluded from total Q&A latency.")
    print("The support audit is model-based, not a guarantee of correctness.")


if __name__ == "__main__":
    main()
