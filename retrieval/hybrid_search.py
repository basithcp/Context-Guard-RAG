from __future__ import annotations

import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from langchain_chroma import Chroma

TOKEN_PATTERN = re.compile(
    r"\$?[A-Za-z_][A-Za-z0-9_]*|>>|<<|[.]{1,2}|[^\w\s]",
    re.UNICODE,
)

def tokenize(text : str) -> list[str]:
    """
    Tokenize natural language and retain useful technical identifiers.

    Examples:
        PWD -> PWD
        pwd -> pwd
        $HOME -> $HOME
        >> -> >>
    """
    tokens = TOKEN_PATTERN.findall(text)
    normalized = []

    for token in tokens:
        if token.startswith("$"):
            normalized.append(token)
        elif token in {">", ">>", "<", "<<", "|"}:
            normalized.append(token)
        elif token.isalpha() and token.isupper() and len(token) > 1:
            normalized.append(token)
        else:
            normalized.append(token.lower())

    return normalized



class HybridRetriever:
    """Combine ChromaDB dense retrieval with BM25 keyword retrieval."""

    def __init__(
        self,
        documents: list[Document],
        vector_store: Chroma,
        rank_constant: int = 60,
    ) -> None:
        if not documents:
            raise ValueError("Cannot index an empty document collection")
        
        if rank_constant <= 0:
            raise ValueError("rank_constant must be positive")

        self.documents = documents
        self.vector_store = vector_store
        self.rank_constant = rank_constant

        self.document_ids = [
            doc.metadata["chunk_id"]
            for doc in documents
        ]

        if len(set(self.document_ids)) != len(self.document_ids):
            raise ValueError("Duplicate chunk IDs found.")
        
        self.bm25 = BM25Okapi(
            [tokenize(doc.page_content) for doc in documents]
        )

    def dense_search(
        self, 
        query: str,
        k: int = 5,
    ) -> list[dict]:
        results = self.vector_store.similarity_search_with_score(
            query,
            k=k
        )

        return [
            {
                "document": document,
                "rank": rank,
                "score": float(distance)
            }
            for rank, (document, distance) in enumerate(
                results,
                1
            )
        ]

    def sparse_search(
        self,
        query: str,
        k: int = 5,
    ) -> list[dict]:
        query_tokens = tokenize(query)
        
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key = lambda index: float(scores[index]),
            reverse= True,
        )

        results = []

        for index in ranked_indices:
            score = float(scores[index])

            if score <= 0:
                continue

            results.append(
                {
                    "document":self.documents[index],
                    "rank": len(results) + 1,
                    "score": score,
                }
            )

            if len(results) == k:
                break
        
        return results

    def search(
        self,
        query: str,
        k: int = 5,
        candidate_k: int = 20,
    ) -> list[dict]:
        """
        Retrieve from dense and sparse search, then fuse ranks with RRF.
        """
        dense_results = self.dense_search(query, k=candidate_k)
        sparse_results = self.sparse_search(query, k=candidate_k)

        fused: dict[str, dict] = {}

        def add_result(
            result: dict,
            source: str,
        ) -> None:
            document = result["document"]
            chunk_id = document.metadata["chunk_id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "document": document,
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "bm25_rank": None,
                    "dense_distance": None,
                    "bm25_score": None
                }

            item = fused[chunk_id]

            item["rrf_score"] += 1.0 / (
                self.rank_constant + result["rank"]
            )

            if source == "dense":
                item["dense_rank"] = result["rank"]
                item["dense_distance"] = result["score"]
            else:
                item["bm25_rank"] = result["rank"]
                item["bm25_score"] = result["score"]

        for result in dense_results:
            add_result(result, "dense")
        
        for result in sparse_results:
            add_result(result, "bm25")

        ranked = sorted(
            fused.values(),
            key= lambda item: item["rrf_score"],
            reverse=True, 
        )

        return ranked[:k]