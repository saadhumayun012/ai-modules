# app/utils/validators.py
"""File validation utilities."""

from app.core.constants import ALLOWED_DOCX_CONTENT_TYPES
from app.core import settings


def validate_docx_file(
    filename: str,
    content_type: str | None,
    file_bytes: bytes,
) -> tuple[bool, str | None]:
    """
    Validate a DOCX file.
    
    Returns:
        (is_valid, error_message)
    """
    filename_lower = (filename or "").lower()
    content_type_lower = (content_type or "").lower()
    
    # Check extension
    if not filename_lower.endswith(".docx"):
        return False, "Only DOCX file is supported"
    
    # Check content type
    if content_type_lower and content_type_lower not in ALLOWED_DOCX_CONTENT_TYPES:
        return False, "Invalid file type. Please upload a .docx file."
    
    # Check file is not empty
    if not file_bytes:
        return False, "Uploaded file is empty"
    
    # Check file size
    max_bytes = settings.file_max_bytes
    if len(file_bytes) > max_bytes:
        max_mb = max_bytes // (1024 * 1024)
        return False, f"File too large. Max allowed size is {max_mb} MB."
    
    return True, None
