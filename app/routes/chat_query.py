import logging
import re

from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.core import settings, client
from app.services import embed_chunks
from app.utils import (
    expand_query,
    extract_keywords,
    deduplicate_chunks,
    clean_response,
    rerank_by_relevance,
    assemble_context,
)

logger = logging.getLogger(__name__)

chat_client = OpenAI(
    api_key=settings.api_key,
    base_url=settings.base_url,
)

router = APIRouter()


def _clean_text(value: str) -> str:
    return value.strip()


def _is_collection_missing_error(exc: Exception) -> bool:
    # Check if error is due to missing Qdrant collection
    message = str(exc).lower()
    return "not found: collection" in message or "doesn't exist" in message


def _is_trivial_query(query: str) -> bool:
    """Check if query is too trivial (less than 2 words)."""
    words = query.strip().split()
    return len(words) < 2


def _format_chatbot_response(response_text: str) -> str:
    """Normalize model output into a natural chatbot-style response."""
    cleaned = clean_response(response_text, preserve_formatting=False)
    if not cleaned:
        return ""

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""

    normalized_parts: list[str] = []
    for line in lines:
        line = re.sub(r"^\s*(?:[-•*]+|\d+[.)])\s*", "", line)
        line = re.sub(r"\(\s*\d+\s*\.?\s*\)", "", line)
        line = re.sub(r"\s+", " ", line).strip(" ,;:-")
        if line:
            normalized_parts.append(line)

    if not normalized_parts:
        return ""

    if len(normalized_parts) == 1:
        result = normalized_parts[0]
    else:
        result = "; ".join(normalized_parts)

    result = re.sub(r"\s+", " ", result).strip()
    if result and result[-1] not in ".!?":
        result += "."
    return result


class ChatRequest(BaseModel):
    document_id: str = Field(..., min_length=1, max_length=128)
    query: str = Field(..., min_length=3, max_length=2000)

    @field_validator("document_id", "query")
    @classmethod
    def strip_and_validate(cls, value: str) -> str:
        # Strip whitespace and reject empty strings after stripping
        cleaned = _clean_text(value)
        if not cleaned:
            raise ValueError("Must not be empty")
        return cleaned
    
    @field_validator("query")
    @classmethod
    def validate_query_quality(cls, value: str) -> str:
        """Ensure query is meaningful, not just a trivial input."""
        if _is_trivial_query(value):
            raise ValueError("Query is too vague or trivial. Please provide a more specific question.")
        return value


@router.post("/chat")
async def chat_query(request: ChatRequest):
    """RAG chat: query expansion → vector search → dedup → rerank → LLM → response cleaning."""
    normalized_document_id = request.document_id.strip().lower()
    retrieval_top_k = settings.retrieval_top_k
    retrieval_score_threshold = settings.retrieval_score_threshold
    retrieval_candidate_k = max(retrieval_top_k * 2, retrieval_top_k)

    # Generate query variations for better retrieval
    query_variations = [request.query]
    if settings.query_expansion_enabled:
        query_variations = expand_query(request.query)
        logger.info(
            "Query expansion generated %d variations for document_id=%s",
            len(query_variations),
            normalized_document_id,
        )

    # Collect results from original query and variations
    all_results_points = []
    all_query_keywords = extract_keywords(request.query)

    for query_var in query_variations:
        try:
            query_embedding = await run_in_threadpool(embed_chunks, [query_var])
            query_vector = query_embedding[0]
        except Exception as exc:
            logger.exception(
                "Query embedding failed for document_id=%s, query_var=%s: %s",
                normalized_document_id,
                query_var,
                exc,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Query embedding failed",
            )

        try:
            results = client.query_points(
                collection_name=settings.qdrant_collection_name,
                query=query_vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=normalized_document_id),
                        )
                    ]
                ),
                limit=retrieval_candidate_k,
                with_payload=True,
            )
            all_results_points.extend(results.points)
        except UnexpectedResponse as exc:
            if _is_collection_missing_error(exc):
                logger.warning(
                    "Collection missing for document_id=%s. Document not indexed.", normalized_document_id
                )
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Document not found. Please index the document first.",
                )
            
            logger.exception(
                "Vector search failed for document_id=%s: %s", normalized_document_id, exc
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Vector search failed",
            )
        except Exception as exc:
            logger.exception("Vector search failed for document_id=%s: %s", normalized_document_id, exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Vector search failed",
            )

    if not all_results_points:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No relevant content found in this document",
        )

    # Filter by semantic score threshold
    points = [p for p in all_results_points if (p.score or 0.0) >= retrieval_score_threshold]

    # If enforce_threshold is True, reject low-scoring results instead of falling back
    if not points:
        if settings.retrieval_enforce_threshold:
            logger.warning(
                "No points met threshold for document_id=%s with threshold=%.2f",
                normalized_document_id,
                retrieval_score_threshold,
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No relevant content found in this document (confidence threshold not met)",
            )
        else:
            # Fallback: use all points but sort carefully
            points = list(all_results_points)

    # Deduplicate similar chunks by text similarity
    if settings.retrieval_deduplication_enabled and len(points) > 1:
        texts = [str((p.payload or {}).get("text", "")) for p in points]
        unique_texts = deduplicate_chunks(
            texts, similarity_threshold=settings.retrieval_deduplication_threshold
        )
        unique_text_set = set(unique_texts)
        points = [p for p in points if str((p.payload or {}).get("text", "")) in unique_text_set]

    # Re-rank by keyword overlap and semantic score
    points_with_data = [
        {
            "point": p,
            "text": str((p.payload or {}).get("text", "")),
            "score": float(p.score or 0.0),
            "payload": p.payload or {},
        }
        for p in points
    ]

    ranked = rerank_by_relevance(
        points_with_data,
        all_query_keywords,
        use_similarity=settings.query_expansion_use_similarity,
        query_text=request.query,
    )

    # Select top-K chunks after re-ranking by relevance
    top_points = ranked[:retrieval_top_k]

    if not top_points:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No relevant content found in this document",
        )

    context_chunks = [{"text": p["text"]} for p in top_points]

    # Assemble context with optional deduplication and length limiting
    context, chunks_used = assemble_context(
        context_chunks,
        max_context_length=settings.retrieval_max_context_length,
        remove_duplicates=settings.retrieval_deduplication_enabled,
        duplicate_threshold=settings.retrieval_deduplication_threshold,
    )

    if not context:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No usable context found in retrieved chunks",
        )

    logger.debug(
        "Using %d chunks (%.0f%% of top_k) for context for document_id=%s",
        chunks_used,
        (chunks_used / retrieval_top_k * 100) if retrieval_top_k > 0 else 0,
        normalized_document_id,
    )

    # Chatbot-focused prompt: natural, conversational, and grounded in the document only.
    system_prompt = f"""You are a helpful document chatbot. Answer the user's question using only the provided document context.
Style rules:
- Write in clear, natural, conversational language.
- Prefer a short explanation or compact paragraph.
- Do not copy awkward numbering such as "1. ." or "(1.)".
- Do not use raw bullet lists unless the user explicitly asks for a list.
- If the context contains several related items, rewrite them naturally so they read like a proper chatbot response.
- If the document does not contain enough information, say that clearly and politely.
- Do not invent facts or add information outside the context.

Document Context:
{context}
"""

    # Call LLM with context and query
    try:
        response = chat_client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.query},
            ],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    except Exception as exc:
        logger.exception("LLM service error for document_id=%s: %s", normalized_document_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LLM service error",
        )

    content = response.choices[0].message.content or "I could not find relevant information in the document"

    # Normalize the reply so it feels like a chatbot, not extracted notes.
    content = _format_chatbot_response(content)
    if not content:
        content = "I could not find relevant information in the document."

    return {
        "document_id": normalized_document_id,
        "response": content,
    }