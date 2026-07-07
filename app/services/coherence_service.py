from typing import cast

import numpy as np

from app.core import settings, get_embedding_model
from app.core import SectionData, SentenceIssue, ParagraphIssue
from app.utils import split_into_sentences

SENTENCE_THRESHOLD = float(settings.coherence_sentence_threshold)
PARAGRAPH_THRESHOLD = float(settings.coherence_paragraph_threshold)
WINDOW_SIZE = int(settings.coherence_sentence_window)
MIN_SENTENCE_WORDS = int(settings.coherence_min_sentence_words)


def embed_sentences(sentences: list[str]) -> np.ndarray:
    embedding_model = get_embedding_model()
    vectors = list(embedding_model.embed(sentences))
    if not vectors:
        return np.empty((0, 0), dtype=np.float32)
    return np.array(vectors, dtype=np.float32)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    norm1 = float(np.linalg.norm(vec1))
    norm2 = float(np.linalg.norm(vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def _preview(text: str, max_words: int = 10) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def analyze_coherence(sections: list[SectionData]) -> dict:
    issues: list[dict] = []

    if WINDOW_SIZE < 1:
        raise ValueError("coherence_sentence_window must be >= 1")

    section_sentence_data: list[list[tuple[int, str]]] = []
    for section in sections:
        section_sents: list[tuple[int, str]] = []
        for para_idx, paragraph in enumerate(section.get("paragraphs", [])):
            for sentence in split_into_sentences(paragraph, min_words=MIN_SENTENCE_WORDS):
                section_sents.append((para_idx, sentence))
        section_sentence_data.append(section_sents)

    all_texts = [s for section in section_sentence_data for (_, s) in section]
    if not all_texts:
        return {"total_issues": 0, "issues": []}

    all_embeddings = embed_sentences(all_texts)
    if all_embeddings.size == 0:
        return {"total_issues": 0, "issues": []}

    # Phase 3: Process each section using its slice of embeddings
    offset = 0
    for section_idx, section in enumerate(sections):
        heading = section.get("heading", "Untitled")
        section_sents = section_sentence_data[section_idx]
        n = len(section_sents)
        if n == 0:
            continue

        section_embeddings = all_embeddings[offset : offset + n]
        offset += n

        # Group sentences and embeddings by paragraph
        para_sentence_map: list[tuple[int, list[str]]] = []
        para_emb_list: list[np.ndarray] = []
        emb_idx = 0

        for para_idx, paragraph in enumerate(section.get("paragraphs", [])):
            sentences = split_into_sentences(paragraph, min_words=MIN_SENTENCE_WORDS)
            if not sentences:
                continue
            para_sentence_map.append((para_idx, sentences))
            para_emb = section_embeddings[emb_idx : emb_idx + len(sentences)]
            para_emb_list.append(para_emb)
            emb_idx += len(sentences)

        # Sentence-level coherence
        para_vectors: list[tuple[int, np.ndarray]] = []

        for pg_idx, (para_idx, sentences) in enumerate(para_sentence_map):
            para_emb = para_emb_list[pg_idx]
            m = len(sentences)

            flagged_indices: set[int] = set()

            for i in range(m):
                if i in flagged_indices:
                    continue

                window_scores: list[tuple[int, float]] = []

                for j in range(i + 1, min(i + WINDOW_SIZE + 1, m)):
                    score = cosine_similarity(para_emb[i], para_emb[j])
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

            paragraph_vector = np.mean(para_emb, axis=0)
            para_vectors.append((para_idx, paragraph_vector))

        # Paragraph-level coherence
        for i in range(len(para_vectors) - 1):
            current_idx, current_vec = para_vectors[i]
            next_idx, next_vec = para_vectors[i + 1]
            score = cosine_similarity(current_vec, next_vec)

            if score < PARAGRAPH_THRESHOLD:
                # Find the paragraph texts for context
                _, curr_sentences = para_sentence_map[i]
                _, next_sentences = para_sentence_map[i + 1]

                paragraph_issue: ParagraphIssue = {
                    "heading": heading,
                    "level": "paragraph",
                    "location": f"Paragraph {current_idx + 1} -> {next_idx + 1}",
                    "score": round(score, 2),
                    "paragraph_1": _preview(curr_sentences[0]) if curr_sentences else "",
                    "paragraph_2": _preview(next_sentences[0]) if next_sentences else "",
                }
                issues.append(paragraph_issue)

    return {
        "total_issues": len(issues),
        "issues": cast(list[dict], issues),
    }
