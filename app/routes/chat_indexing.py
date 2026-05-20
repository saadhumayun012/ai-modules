import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status
from fastapi.concurrency import run_in_threadpool

from app.core import settings
from app.utils.validators import validate_docx_file
from app.services.indexing_service import (
    extract_text_from_docx,
    chunk_content,
    embed_chunks,
    store_chunks,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/indexing")
async def index_document(
    document_id: str = Form(...),
    file: UploadFile = File(...),
):
    document_id = document_id.strip().lower()
    if not document_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="document_id is required",
        )

    filename = file.filename or ""
    content_type = file.content_type or ""
    file_bytes = await file.read()
    
    # Check file: extension, content-type, size limits
    is_valid, error_msg = validate_docx_file(filename, content_type, file_bytes)
    if not is_valid:
        if "too large" in (error_msg or "").lower():
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=error_msg,
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg,
        )

    # Extract text and headings from DOCX (paragraphs + tables)
    try:
        sections_data = await run_in_threadpool(extract_text_from_docx, file_bytes)
    except Exception as exc:
        logger.exception("DOCX parsing failed for document_id=%s: %s", document_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not extract text from DOCX",
        )

    if not sections_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No readable text found in DOCX",
        )

    # Split content into overlapping chunks
    try:
        chunks_data = await run_in_threadpool(chunk_content, sections_data, document_id)
    except Exception as exc:
        logger.exception("Chunking failed for document_id=%s: %s", document_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chunk creation failed",
        )

    if not chunks_data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No chunks were created",
        )

    # Generate embeddings for all chunks
    try:
        embeddings = await run_in_threadpool(
            embed_chunks,
            [chunk["text"] for chunk in chunks_data],
        )
    except Exception as exc:
        logger.exception("Embedding failed for document_id=%s: %s", document_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embedding generation failed",
        )

    # Store chunks with embeddings in Qdrant
    try:
        await run_in_threadpool(store_chunks, chunks_data, embeddings, document_id)
    except ValueError as exc:
        logger.exception("Storage validation failed for document_id=%s: %s", document_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Vector storage failed for document_id=%s: %s", document_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Vector storage failed",
        )

    return {
        "message": "Indexing done",
        "document_id": document_id,
        "chunks_created": len(chunks_data),
    }