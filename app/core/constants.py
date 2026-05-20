# app/core/constants.py
"""Shared constants across the application."""

# Heading styles that indicate document sections
HEADING_STYLES = [
    "Thesis Section",
    "Thesis Heading",
    "Thesis Subheading",
    "Heading 1",
    "Heading 2",
    "Heading 3",
    "Heading 4",
    "Heading 5",
    "Heading 6",
    "Header",
    "Title",
    "Sub Header",
]
HEADING_STYLES_LOWER = {style.lower() for style in HEADING_STYLES}

# Styles to skip (TOC, figures, captions, etc.)
SKIP_STYLES = [
    "toc 1",
    "toc 2",
    "toc 3",
    "toc 4",
    "table of figures",
    "thesis figure caption",
    "thesis table caption",
    "thesis quote",
    "thesis figure legend",
    "thesis figure",
    "cover title",
    "cover heading",
]
SKIP_STYLES_LOWER = {style.lower() for style in SKIP_STYLES}

# Headings to skip (References, Mockups, etc.)
SKIP_HEADINGS = ["References", "Gantt chart", "Mockups", "Plagiarism Report", "Diagrams", "Appendix"]
SKIP_HEADINGS_LOWER = {h.lower() for h in SKIP_HEADINGS}

# Allowed content types for DOCX uploads
ALLOWED_DOCX_CONTENT_TYPES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/wps-office.docx",
    "application/octet-stream",  # some clients send this for docx
}

# Utility Functions
def is_skip_heading(text: str) -> bool:
    """Check if text matches any skip heading (References, Mockups, etc.)."""
    text_lower = text.strip().lower()
    return any(skip in text_lower for skip in SKIP_HEADINGS_LOWER)
