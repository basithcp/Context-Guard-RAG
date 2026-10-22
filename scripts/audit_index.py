from collections import defaultdict
from pathlib import Path

import chromadb
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from retrieval.vector_store import CHROMA_DIR, COLLECTION_NAME


def main():
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    data = collection.get(
        include=["metadatas"]
    )

    documents = defaultdict(
        lambda: {
            "chunks": 0,
            "pages": set(),
            "total_pages": set(),
        }
    )

    for metadata in data["metadatas"] or []:
        if not metadata:
            continue

        document_id = metadata.get(
            "document_id", "UNKNOWN"
        )
        source = metadata.get(
            "source", "UNKNOWN"
        )

        key = (document_id, source)
        item = documents[key]

        item["chunks"] += 1

        page = metadata.get("page_number")
        if isinstance(page, int):
            item["pages"].add(page)

        total = metadata.get("total_pages")
        if isinstance(total, int):
            item["total_pages"].add(total)

    print("INDEX AUDIT")
    print("=" * 65)
    print("Collection:", COLLECTION_NAME)
    print("Total chunk records:", collection.count())
    print("Distinct document identities:", len({
        document_id
        for document_id, _ in documents
    }))

    for (document_id, source), item in sorted(
        documents.items(),
        key=lambda entry: entry[0][1].lower(),
    ):
        print(f"\nSource: {source}")
        print(f"Document ID prefix: {document_id[:16]}")
        print(f"Chunks: {item['chunks']}")
        print(f"Pages represented: {len(item['pages'])}")
        print(f"Reported total pages: {sorted(item['total_pages'])}")

    print("\nAudit completed. No records were modified.")


if __name__ == "__main__":
    main()