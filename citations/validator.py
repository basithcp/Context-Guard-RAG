from __future__ import annotations

from generation.answer_generator import ABSTENTION_TEXT


class CitationValidationError(ValueError):
    """Raised when an answer contains invalid citation references."""


def validate_answer(
    payload: dict,
    evidence: list[dict],
) -> dict:
    """
    Validate the answer structure and its references to retrieved evidence.

    This checks citation provenance and schema, not factual entailment.
    """
    if not isinstance(payload, dict):
        raise CitationValidationError(
            "The answer must be a JSON object."
        )

    abstain = payload.get("abstain")
    claims = payload.get("claims")

    if not isinstance(abstain, bool):
        raise CitationValidationError(
            "The answer must contain a boolean 'abstain' field."
        )

    if not isinstance(claims, list):
        raise CitationValidationError(
            "The answer must contain a 'claims' list."
        )

    if abstain:
        if claims:
            raise CitationValidationError(
                "An abstaining answer cannot contain factual claims."
            )

        return {
            "abstain": True,
            "claims": [],
        }

    if not claims:
        raise CitationValidationError(
            "A non-abstaining answer must contain at least one claim."
        )

    evidence_by_id = {
        item["source_id"]: item
        for item in evidence
    }

    validated_claims = []

    for index, claim in enumerate(claims, start=1):
        if not isinstance(claim, dict):
            raise CitationValidationError(
                f"Claim {index} must be a JSON object."
            )

        text = claim.get("text")
        source_ids = claim.get("source_ids")

        if not isinstance(text, str) or not text.strip():
            raise CitationValidationError(
                f"Claim {index} has no valid text."
            )

        if (
            not isinstance(source_ids, list)
            or not source_ids
            or not all(
                isinstance(source_id, str)
                and source_id.strip()
                for source_id in source_ids
            )
        ):
            raise CitationValidationError(
                f"Claim {index} must cite at least one evidence ID."
            )

        unknown_ids = [
            source_id
            for source_id in source_ids
            if source_id not in evidence_by_id
        ]

        if unknown_ids:
            raise CitationValidationError(
                f"Claim {index} references unknown evidence IDs: "
                f"{unknown_ids}"
            )

        # Remove duplicate IDs while preserving their order.
        unique_ids = list(dict.fromkeys(source_ids))

        validated_claims.append(
            {
                "text": text.strip(),
                "source_ids": unique_ids,
            }
        )

    return {
        "abstain": False,
        "claims": validated_claims,
    }


def render_answer(
    validated_answer: dict,
    evidence: list[dict],
) -> str:
    """
    Render validated claims with filename and physical PDF page citations.
    """
    if validated_answer["abstain"]:
        return ABSTENTION_TEXT

    evidence_by_id = {
        item["source_id"]: item
        for item in evidence
    }

    rendered_claims = []

    for claim in validated_answer["claims"]:
        citations = []
        seen = set()

        for source_id in claim["source_ids"]:
            item = evidence_by_id[source_id]

            source = item["source"]
            page_number = item["page_number"]

            key = (source, page_number)

            if key in seen:
                continue

            seen.add(key)
            citations.append(f"{source}, p. {page_number}")

        citation_text = "; ".join(
            f"[{citation}]"
            for citation in citations
        )

        rendered_claims.append(
            f"{claim['text']} {citation_text}"
        )

    return "\n\n".join(rendered_claims)