import json
import re
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET = ROOT / "evaluation" / "answer_eval_45.json"
RAW_DIR = ROOT / "data" / "raw"

STOPWORDS = {
    "what", "which", "where", "when", "does", "this", "that",
    "from", "with", "into", "your", "their", "they", "them",
    "have", "been", "being", "will", "would", "could", "should",
    "about", "according", "document", "question", "answer",
    "command", "commands", "used", "uses", "using", "does",
    "exact", "currently", "current", "provided", "provide",
    "indicate", "indicates", "mean", "means", "book", "page",
}

def locate_pdf(name):
    # Prefer the exact original filename if it exists.
    exact = list(RAW_DIR.rglob(name))
    if exact:
        return exact[0]

    # Otherwise resolve a uniquely named hash-suffixed upload.
    stem = Path(name).stem
    candidates = [
        candidate
        for candidate in RAW_DIR.rglob("*.pdf")
        if candidate.stem == stem
        or candidate.stem.startswith(stem + "__")
    ]
    return candidates[0] if len(candidates) == 1 else None


def tokens(text):
    return {
        word for word in re.findall(r"[a-zA-Z0-9_$.-]+", text.lower())
        if len(word) >= 3 and word not in STOPWORDS
    }


def relevant_sentences(page_text, question, reference):
    terms = tokens(question) | tokens(reference)
    parts = re.split(r"(?<=[.!?])\s+|\n+", page_text)
    ranked = []

    for index, sentence in enumerate(parts):
        clean = " ".join(sentence.split())
        if len(clean) < 25:
            continue
        words = tokens(clean)
        score = len(words & terms)
        if score:
            ranked.append((score, index, clean))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = []
    seen = set()

    for score, index, sentence in ranked:
        if sentence in seen:
            continue
        seen.add(sentence)
        selected.append(sentence[:400])
        if len(selected) == 3:
            break

    return selected


def main():
    questions = json.loads(DATASET.read_text(encoding="utf-8"))
    cache = {}
    errors = 0

    for item in questions:
        print("\n" + "=" * 78)
        print(f"{item['id']} [{item['answerability']}] {item['question']}")
        print("REFERENCE:", item["reference_answer"])

        if not item["gold_sources"]:
            print("GOLD EVIDENCE: none (check whether abstention is justified)")
            continue

        for source in item["gold_sources"]:
            name = source["source"]
            page_number = source["page_number"]

            if name not in cache:
                path = locate_pdf(name)
                if path is None:
                    print(f"ERROR: cannot resolve PDF {name}")
                    errors += 1
                    continue
                with pymupdf.open(path) as pdf:
                    cache[name] = [p.get_text("text") for p in pdf]

            pages = cache[name]
            print(f"\nSOURCE: {name}, physical PDF page {page_number}")

            if not 1 <= page_number <= len(pages):
                print("ERROR: invalid page number")
                errors += 1
                continue

            excerpts = relevant_sentences(
                pages[page_number - 1],
                item["question"],
                item["reference_answer"],
            )

            if not excerpts:
                print("No matching sentence selected; review the full page.")
            else:
                for excerpt in excerpts:
                    print(f"  - {excerpt}")

    print("\n" + "=" * 78)
    print(f"Questions reviewed: {len(questions)}")
    print(f"Source-resolution/page errors: {errors}")
    print("Excerpt matching is only a review aid, not proof of entailment.")


if __name__ == "__main__":
    main()
