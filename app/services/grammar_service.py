# services/grammar_service.py
import difflib
import re
from typing import Any

import httpx

from app.core import settings
from app.utils import get_nlp, split_into_sentences

_CITATION_PATTERNS = [
    r"\([A-Z][a-zA-Z\s]+,?\s*\d{4}[a-z]?\)",
    r"\[\d+(?:,\s*\d+)*\]",
    r"[A-Z][a-z]+\s+et\s+al\.\s*\(\d{4}\)",
]


def handle_citations(text: str) -> tuple[str, list[str]]:
    citations: list[str] = []

    def replacer(match: re.Match) -> str:
        placeholder = f"CITATION{len(citations)}"
        citations.append(match.group())
        return placeholder

    updated = text
    for pattern in _CITATION_PATTERNS:
        updated = re.sub(pattern, replacer, updated)

    return updated, citations


def restore_citations(text: str, citations: list[str]) -> str:
    restored = text
    for i, citation in enumerate(citations):
        restored = restored.replace(f"CITATION{i}", citation)
    return restored


def _fast_check(sentence: str) -> bool:
    text = sentence.strip()

    if len(text.split()) < settings.grammar_min_sentence_words:
        return False
    if re.match(r"^[\u2022\u2219\-*]\s+", text):
        return False
    if re.match(r"^[A-Z]{1,5}-\d+[\.\:]\s*", text):
        return False
    if re.search(r"https?://", text):
        return False

    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.5:
        return False

    return True


def highlight_diff(original: str, corrected: str) -> tuple[str, list[dict[str, str]]]:
    orig_words = original.split()
    corr_words = corrected.split()
    matcher = difflib.SequenceMatcher(None, orig_words, corr_words)

    highlighted: list[str] = []
    changes: list[dict[str, str]] = []

    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode == "equal":
            highlighted.extend(corr_words[j1:j2])
            continue

        original_chunk = " ".join(orig_words[i1:i2])
        corrected_chunk = " ".join(corr_words[j1:j2])

        if opcode == "replace":
            highlighted.append(f"[{original_chunk} -> {corrected_chunk}]")
            changes.append(
                {
                    "type": "replace",
                    "original": original_chunk,
                    "corrected": corrected_chunk,
                }
            )
        elif opcode == "delete":
            highlighted.append(f"[{original_chunk} -> REMOVED]")
            changes.append(
                {"type": "delete", "original": original_chunk, "corrected": ""}
            )
        elif opcode == "insert":
            highlighted.append(f"[ADDED -> {corrected_chunk}]")
            changes.append(
                {"type": "insert", "original": "", "corrected": corrected_chunk}
            )

    return " ".join(highlighted), changes


def build_grammar_items(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    candidates: list[tuple[str, str]] = []

    for section in sections:
        heading = section.get("heading", "Document")
        for paragraph in section.get("paragraphs", []):
            for sentence in split_into_sentences(
                paragraph,
                min_words=settings.grammar_min_sentence_words,
            ):
                if not _fast_check(sentence):
                    continue
                candidates.append((heading, sentence))

    if not candidates:
        return items

    nlp = get_nlp()
    texts = [s for _, s in candidates]
    docs = nlp.pipe(texts, batch_size=64)

    for (heading, sentence), doc in zip(candidates, docs):
        if not any(token.pos_ == "VERB" for token in doc):
            continue
        clean, citations = handle_citations(sentence)
        items.append({
            "heading": heading,
            "original": sentence,
            "clean": clean,
            "citations": citations,
        })

    return items


def apply_predictions(
    items: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    if len(items) != len(predictions):
        raise ValueError("Prediction count does not match input count.")

    corrections: list[dict[str, Any]] = []

    for item, prediction in zip(items, predictions):
        corrected_clean = str(prediction.get("corrected", "")).strip()
        if not corrected_clean:
            continue

        confidence = float(prediction.get("confidence", 1.0))
        if confidence < confidence_threshold:
            continue

        corrected_full = restore_citations(corrected_clean, item.get("citations", []))
        if corrected_full.strip() == str(item.get("original", "")).strip():
            continue

        highlighted, changes = highlight_diff(item["original"], corrected_full)
        corrections.append(
            {
                "heading": item["heading"],
                "original": item["original"],
                "corrected": corrected_full,
                "highlighted": highlighted,
                "changes": changes,
                "confidence": round(confidence, 4),
            }
        )

    return corrections


async def check_structured_grammar(filename: str, sections: list[dict]):
    """Process sections locally, call Colab inference, and return corrections."""
    timeout = httpx.Timeout(
        connect=settings.grammar_api_connect_timeout,
        read=settings.grammar_api_timeout,
        write=20.0,
        pool=5.0,
    )

    items = build_grammar_items(sections)
    if not items:
        return {
            "filename": filename,
            "total_sections": len(sections),
            "total_corrections": 0,
            "corrections": [],
        }

    batch_size = max(1, int(settings.grammar_batch_size))
    predictions: list[dict] = []

    async with httpx.AsyncClient(timeout=timeout) as client:
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            response = await client.post(
                f"{settings.grammar_api_url}",
                json={
                    "texts": [item["clean"] for item in batch],
                },
            )
            response.raise_for_status()
            payload = response.json()
            batch_predictions = payload.get("predictions", [])
            if len(batch_predictions) != len(batch):
                raise ValueError("Inference response count did not match request batch size.")
            predictions.extend(batch_predictions)

    corrections = apply_predictions(
        items,
        predictions,
        confidence_threshold=settings.grammar_confidence_threshold,
    )

    return {
        "filename": filename,
        "total_sections": len(sections),
        "total_corrections": len(corrections),
        "corrections": corrections,
    }