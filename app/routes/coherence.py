import logging
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.services.document_extractor import extract_document_structure
from app.services.coherence_service import analyze_coherence

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/coherence")
async def coherence(file: UploadFile = File(...)):
    """Upload DOCX → analyze coherence at sentence + paragraph level."""
    filename = file.filename or ""

    # Extract document structure with validation
    try:
        structure = await extract_document_structure(
            file,
            include_tables=False 
        )
    except ValueError as exc:
        # Validation error from validate_docx_file
        detail = str(exc)
        status_code = (
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            if "too large" in detail.lower()
            else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(status_code=status_code, detail=detail)
    except Exception as exc:
        logger.exception("DOCX extraction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process document.",
        )

    # Verify structure has content
    if not structure or all(not s.get("paragraphs") for s in structure):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No analyzable text found in document.",
        )

    # Analyze coherence: sentence-to-sentence and paragraph-to-paragraph similarity
    try:
        result = await run_in_threadpool(analyze_coherence, structure)
    except Exception as exc:
        logger.exception("Coherence analysis failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Coherence analysis failed.",
        )

    return {
        "filename": filename,
        **result,
    }