import re
from typing import cast

import numpy as np
from fastembed import TextEmbedding

from app.core import settings
from app.core.constants import HEADING_STYLES_LOWER, SKIP_STYLES_LOWER, SKIP_HEADINGS_LOWER, is_skip_heading
from app.core.models import SectionData, SentenceIssue, ParagraphIssue
from app.core.embedding import get_embedding_model

try:
    import spacy  # type: ignore
except ImportError:  # pragma: no cover
    spacy = None

SENTENCE_THRESHOLD = float(settings.coherence_sentence_threshold)
PARAGRAPH_THRESHOLD = float(settings.coherence_paragraph_threshold)
WINDOW_SIZE = int(settings.coherence_sentence_window)
MIN_SENTENCE_WORDS = int(settings.coherence_min_sentence_words)

_NLP = None
if spacy is not None:
    try:
        _NLP = spacy.blank("en")
        if "sentencizer" not in _NLP.pipe_names:
            _NLP.add_pipe("sentencizer")
    except Exception:
        _NLP = None


def split_into_sentences(paragraph: str) -> list[str]:
    # Split paragraph into sentences using spaCy or regex; filter short sentences
    if _NLP is not None:
        doc = _NLP(paragraph)
        raw_sentences = [s.text.strip() for s in doc.sents]
    else:
        raw_sentences = re.split(r"(?<=[.!?])\s+", paragraph)

    return [s for s in raw_sentences if s and len(s.split()) >= MIN_SENTENCE_WORDS]


def embed_sentences(sentences: list[str]) -> np.ndarray:
    # Generate embeddings for list of sentences
    embedding_model = get_embedding_model()
    vectors = list(embedding_model.embed(sentences))
    if not vectors:
        return np.empty((0, 0), dtype=np.float32)
    return np.array(vectors, dtype=np.float32)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    # Compute normalized dot product between two vectors
    norm1 = float(np.linalg.norm(vec1))
    norm2 = float(np.linalg.norm(vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def analyze_coherence(sections: list[SectionData]) -> dict:
    issues: list[dict] = []

    if WINDOW_SIZE < 1:
        raise ValueError("coherence_sentence_window must be >= 1")

    for section in sections:
        heading = section.get("heading", "Untitled")
        paragraphs = section.get("paragraphs", [])

        if not paragraphs:
            continue

        para_sentence_map: list[tuple[int, list[str]]] = []
        all_sentences: list[str] = []

        for para_idx, paragraph in enumerate(paragraphs):
            sentences = split_into_sentences(paragraph)
            if sentences:
                para_sentence_map.append((para_idx, sentences))
                all_sentences.extend(sentences)

        if not all_sentences:
            continue

        all_embeddings = embed_sentences(all_sentences)
        if all_embeddings.size == 0:
            continue

        paragraph_embeddings: list[tuple[int, np.ndarray]] = []
        offset = 0

        for para_idx, sentences in para_sentence_map:
            n = len(sentences)
            para_embeddings = all_embeddings[offset : offset + n]
            offset += n

            flagged_indices: set[int] = set()

            for i in range(len(sentences)):
                if i in flagged_indices:
                    continue

                window_scores: list[tuple[int, float]] = []

                # Compare sentence i against next WINDOW_SIZE sentences
                for j in range(i + 1, min(i + WINDOW_SIZE + 1, len(sentences))):
                    score = cosine_similarity(para_embeddings[i], para_embeddings[j])
                    window_scores.append((j, score))

                if not window_scores:
                    continue

                max_score = max(score for _, score in window_scores)
                if max_score < SENTENCE_THRESHOLD:
                    worst_j, worst_score = min(window_scores, key=lambda x: x[1])
                    flagged_indices.add(i)
                    flagged_indices.add(worst_j)

                    sentence_issue: SentenceIssue = {
                        "heading": heading,
                        "level": "sentence",
                        "location": f"Paragraph {para_idx + 1}, Sentence {i + 1} -> {worst_j + 1}",
                        "score": round(worst_score, 2),
                        "sentence_1": sentences[i],
                        "sentence_2": sentences[worst_j],
                    }
                    issues.append(sentence_issue)

            paragraph_vector = np.mean(para_embeddings, axis=0)
            paragraph_embeddings.append((para_idx, paragraph_vector))

        # Compare adjacent paragraph vectors (mean of sentence embeddings)
        for i in range(len(paragraph_embeddings) - 1):
            current_idx, current_vec = paragraph_embeddings[i]
            next_idx, next_vec = paragraph_embeddings[i + 1]

            score = cosine_similarity(current_vec, next_vec)
            if score < PARAGRAPH_THRESHOLD:
                paragraph_issue: ParagraphIssue = {
                    "heading": heading,
                    "level": "paragraph",
                    "location": f"Paragraph {current_idx + 1} -> {next_idx + 1}",
                    "score": round(score, 2),
                }
                issues.append(paragraph_issue)

    return {
        "total_issues": len(issues),
        "issues": cast(list[dict], issues),
    }