from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.embeddings import get_embeddings
from retrieval.vector_store import get_vector_store

MODEL_NAME = "openai/gpt-oss-20b"
TOP_K = 5
CANDIDATE_K = 10
RRF_CONSTANT = 60

def rewrite_query(llm: ChatGroq, query: str) -> list[str]:
    """Generate alternative search queries without answering the question."""

    response = llm.invoke([
        (
            "system",
            """
You are a query-expansion component for a technical-document
retrieval system.

Generate exactly three complementary search queries for the
user's question.

Requirements:
1. One natural-language paraphrase preserving the original intent.
2. One identifier-aware query. Consider canonical technical
   identifiers, names, acronyms, commands, variables, and symbols
   that are reasonably associated with the question.
3. One concise keyword-oriented query using important technical
   terms likely to appear in documentation.

Avoid producing three nearly identical paraphrases.
Preserve important distinctions between related technical concepts.
Do not answer the user's question.
Do not claim to have read the source document.
Do not invent document-specific facts.
Return a JSON array of exactly three strings.
Return JSON only, without Markdown fences.
""".strip(),
        ),
        (
            "human",
            f"Rewrite this search query:\n{query}",
        ),
    ])

    raw = str(response.content).strip()

    # Accommodate responses that unexpectedly include Markdown fences.
    raw = re.sub(
        r"^\s*```(?:json)?\s*|\s*```\s*$",
        "",
        raw,
        flags=re.IGNORECASE,
    )

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Model did not return valid JSON: {raw}"
        ) from exc
    
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON array of search queries.")
    
    queries = [
        item.strip()
        for item in parsed
        if isinstance(item, str) and item.strip() 
    ]

    if len(queries) != 3:
        raise ValueError(
            f"Expected three non-empty query strings; received {len(queries)}."
        )
    
    return queries

def retrieve_and_fuse(
    vector_store,
    queries: list[str],
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Retrieve candidates per query and combine their ranks using RRF.
    """
    fused: dict[str, dict] = {}

    for query in queries:
        results = vector_store.similarity_search_with_score(
            query,
            k=CANDIDATE_K,
        )

        for rank, (document, distance) in enumerate(results, start=1):
            chunk_id = document.metadata["chunk_id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "document": document,
                    "rrf_score": 0.0,
                    "best_distance": float(distance),
                }

            fused[chunk_id]["rrf_score"] += (
                1.0 / (RRF_CONSTANT + rank)
            )

            fused[chunk_id]["best_distance"] = min(
                fused[chunk_id]["best_distance"],
                float(distance),
            )

    ranked = sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    return ranked[:top_k]

def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    import os

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your project's .env file."
        )

    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=0,
    )

    query = (
        "Which environment variable represents "
        "the current working directory?"
    )

    print("Original question:")
    print(query)

    rewrites = rewrite_query(llm, query)

    queries = list(dict.fromkeys([query, *rewrites]))

    print("\nSearch queries:")
    for index, item in enumerate(queries, start=1):
        print(f"{index}. {item}")

    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)

    # Compare the original query against multi-query retrieval.
    original_results = vector_store.similarity_search_with_score(
        query,
        k=TOP_K,
    )

    print("\nORIGINAL DENSE RETRIEVAL")

    for rank, (document, distance) in enumerate(
        original_results,
        start=1,
    ):
        print(
            f"Rank {rank} | "
            f"Page {document.metadata['page_number']} | "
            f"Distance {distance:.4f}"
        )

    fused_results = retrieve_and_fuse(vector_store, queries)

    print("\nMULTI-QUERY DENSE RETRIEVAL")

    for rank, item in enumerate(fused_results, start=1):
        document = item["document"]

        print(
            f"Rank {rank} | "
            f"Page {document.metadata['page_number']} | "
            f"RRF {item['rrf_score']:.5f} | "
            f"Best distance {item['best_distance']:.4f}"
        )

        print(
            "  "
            + document.page_content[:220].replace("\n", " ")
        )

    target_page = 161

    original_pages = [
        document.metadata["page_number"]
        for document, _ in original_results
    ]

    fused_pages = [
        item["document"].metadata["page_number"]
        for item in fused_results
    ]

    print("\nEVALUATION")
    print(f"Target page: {target_page}")
    print(f"Original found page: {target_page in original_pages}")
    print(f"Multi-query found page: {target_page in fused_pages}")

    if target_page not in fused_pages:
        print(
            "The rewritten queries did not retrieve the target "
            "page within the fused top results."
        )


if __name__ == "__main__":
    main()