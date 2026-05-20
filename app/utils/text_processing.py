"""Text processing utilities for query expansion, deduplication, and response cleaning."""

import re
from difflib import SequenceMatcher
from typing import Optional


def expand_query(query: str) -> list[str]:
    """
    Expand query with variations for better retrieval.
    
    Returns a list of query variations, with the original query first.
    """
    variations = [query]
    
    # Remove question words from start to create literal search term
    question_words = r"^\b(what|how|why|when|where|which|who|whose|whom|is|are|can|could|would|should|do|does|did)\b\s+"
    simplified = re.sub(question_words, "", query, flags=re.IGNORECASE)
    if simplified and simplified != query:
        variations.append(simplified)
    
    return variations


def deduplicate_chunks(texts: list[str], similarity_threshold: float = 0.7) -> list[str]:
    """Remove near-duplicate chunks based on text similarity."""
    if not texts:
        return []
    
    unique_texts = []
    
    for current_text in texts:
        is_duplicate = False
        
        for existing_text in unique_texts:
            similarity = SequenceMatcher(None, current_text, existing_text).ratio()
            if similarity >= similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_texts.append(current_text)
    
    return unique_texts


def compute_text_similarity(text1: str, text2: str) -> float:
    """Compute text similarity using sequence matching (0-1 scale)."""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def clean_response(response_text: str, preserve_formatting: bool = False) -> str:
    """Clean LLM response: remove markdown, normalize whitespace, preserve lists."""
    if not response_text:
        return ""
    
    # Detect list-like content (numbered or bulleted)
    is_list_like = bool(re.search(r"(?:^|\n)\s*(?:\d+\.|[-•]|\*)\s+", response_text))
    
    if not preserve_formatting:
        # Only remove formatting if not explicitly preserved
        # Remove markdown bold only if it's actual markdown (** word **)
        response_text = re.sub(r"\*\*([^*]+)\*\*", r"\1", response_text)
        
        # Remove markdown italic only if it's actual markdown (* word *)
        response_text = re.sub(r"\*([^*]+)\*", r"\1", response_text)
        
        # Remove markdown list markers at line start (- item) → but keep structure for list detection
        # Only do this if it's NOT list-like content
        if not is_list_like:
            response_text = re.sub(r"(?:^|\n)\s*[-•]\s+", "\n", response_text)
    
    # For list-like content, normalize but preserve line breaks
    if is_list_like:
        # Split into lines and clean each one individually
        lines = response_text.split("\n")
        cleaned_lines = []
        
        for line in lines:
            # Normalize spaces within each line but keep the line
            cleaned_line = " ".join(line.split())
            if cleaned_line:  # Only keep non-empty lines
                cleaned_lines.append(cleaned_line)
        
        return "\n".join(cleaned_lines).strip()
    else:
        # For non-list content, flatten everything to a single paragraph
        # Replace newlines with spaces
        response_text = response_text.replace("\n", " ")
        
        # Normalize multiple spaces to single space
        response_text = " ".join(response_text.split())
        
        return response_text.strip()


def extract_keywords(text: str, min_length: int = 3) -> set[str]:
    """Extract lowercase alphanumeric tokens from text."""
    return {
        token 
        for token in re.findall(r"[a-z0-9]+", text.lower()) 
        if len(token) > min_length
    }


def score_keyword_overlap(query_keywords: set[str], text: str) -> int:
    """Count overlapping keywords between query and text."""
    if not query_keywords:
        return 0
    
    text_keywords = extract_keywords(text)
    return len(query_keywords.intersection(text_keywords))


def rerank_by_relevance(
    chunks: list[dict],
    query_keywords: set[str],
    use_similarity: bool = True,
    query_text: Optional[str] = None
) -> list[dict]:
    """Re-rank chunks by relevance: keyword overlap (primary) > semantic score > text similarity."""
    if not chunks:
        return []
    
    def rank_key(chunk):
        text = chunk.get("text", "")
        
        # Primary: keyword overlap
        keyword_score = score_keyword_overlap(query_keywords, text)
        
        # Secondary: semantic score (from vector search)
        semantic_score = chunk.get("score", 0.0)
        
        # Tertiary: text similarity (if enabled and query provided)
        similarity_score = 0.0
        if use_similarity and query_text:
            similarity_score = compute_text_similarity(query_text, text)
        
        # Return tuple for multi-level sorting (higher is better)
        return (keyword_score, semantic_score, similarity_score)
    
    return sorted(chunks, key=rank_key, reverse=True)


def assemble_context(
    chunks: list[dict],
    max_context_length: Optional[int] = None,
    remove_duplicates: bool = True,
    duplicate_threshold: float = 0.75
) -> tuple[str, int]:
    """Assemble context string from chunks with optional deduplication and length limiting."""
    if not chunks:
        return "", 0
    
    texts = [chunk.get("text", "").strip() for chunk in chunks if chunk.get("text", "").strip()]
    
    if not texts:
        return "", 0
    
    # Remove duplicates if requested
    if remove_duplicates:
        texts = deduplicate_chunks(texts, similarity_threshold=duplicate_threshold)
    
    # Assemble context with length limit
    context_parts = []
    current_length = 0
    
    for text in texts:
        if max_context_length and current_length + len(text) > max_context_length:
            break
        context_parts.append(text)
        current_length += len(text) + 2  # +2 for separators
    
    context = "\n\n".join(context_parts)
    
    return context, len(context_parts)
