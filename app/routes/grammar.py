# routers/grammar.py
from fastapi import APIRouter, HTTPException, UploadFile, File, status
from fastapi.concurrency import run_in_threadpool
import httpx
from app.services import extract_document_structure
from app.services import check_structured_grammar
import logging

router  = APIRouter(prefix="/grammar", tags=["Grammar"])
logger  = logging.getLogger(__name__)


@router.post("/check-document")
async def check_document(file: UploadFile = File(...)):
    """
    Upload DOCX → get grammar corrections with heading + location.
    Only returns sentences that actually have errors.
    """

    # Extract document structure with validation
    try:
        structure = await extract_document_structure(file)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Structure extraction failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to process document.")

    if not structure or all(not s.get("paragraphs") for s in structure):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No analyzable text found. Check document has proper Heading styles.",
        )

    # Call Colab grammar checking API with structured content
    try:
        result = await check_structured_grammar(
            filename=file.filename or "document.docx",
            sections=structure
        )
        return result
    except httpx.ReadTimeout:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={
                "message": "Grammar checking is taking too long.",
                "hint": "Your document is large or the Colab grammar server is slow. Try a smaller file or retry after a moment.",
                "code": "GRAMMAR_TIMEOUT",
            },
        )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "Grammar service is unreachable right now.",
                "hint": "Check whether the Colab server is running and the API URL is correct.",
                "code": "GRAMMAR_UNAVAILABLE",
            },
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "message": "Grammar service returned an error.",
                "hint": str(e),
                "code": "GRAMMAR_UPSTREAM_ERROR",
            },
        )
    except Exception as e:
        logger.exception("Grammar check failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "message": "Grammar check failed unexpectedly.",
                "hint": "Please try again.",
                "code": "GRAMMAR_UNKNOWN_ERROR",
            },
        )