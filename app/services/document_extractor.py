# services/document_extractor.py
import io
import re
from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph as DocxParagraph
from docx.table import Table as DocxTable
from fastapi import UploadFile

from app.core.constants import HEADING_STYLES_LOWER, SKIP_STYLES_LOWER, SKIP_HEADINGS_LOWER, is_skip_heading
from app.core.models import SectionData
from app.utils.validators import validate_docx_file

# ── Helpers ───────────────────────────────────────────────────


def get_heading_level(style_name: str) -> str:
    """Extract heading depth from style name (e.g., 'Heading 1', 'Heading 2')."""
    style_lower = style_name.strip().lower()
    for level in ["heading 1", "heading 2", "heading 3",
                  "heading 4", "heading 5", "heading 6"]:
        if level in style_lower:
            return level.title()   # "Heading 1"
    return "Heading 1"             # fallback for title/header styles


def is_fallback_heading(para) -> tuple[bool, str]:
    """
    Fallback: detect headings when proper Word styles not used.
    Checks formatting — bold (run + style level), font size, ALL CAPS, short lines.
    Returns (is_heading, heading_level)
    """
    text = para.text.strip()

    if not text or len(text) > 120:
        return False, ""

    # Must not end with sentence punctuation
    if text.endswith((',', ';')):
        return False, ""

    # Check runs for formatting
    runs       = [run for run in para.runs if run.text.strip()]
    word_count = len(text.split())

    # Bold check: run level + paragraph style level
    run_bold   = runs and all(run.bold is True for run in runs)
    style_bold = (
        para.style and
        para.style.font.bold is True
    ) if para.style else False
    is_bold = run_bold or style_bold

    # Font size: check runs first, then paragraph style
    font_sizes = [
        run.font.size.pt
        for run in runs
        if run.font.size
    ]
    avg_size   = sum(font_sizes) / len(font_sizes) if font_sizes else 0

    # Also check paragraph style font size
    if not avg_size and para.style and para.style.font.size:
        avg_size = para.style.font.size.pt

    is_all_caps = text.isupper() and len(text) > 3

    # Heuristics: bold+short→Heading 1, large+short→Heading 1, ALL_CAPS→Heading 1, numbered→dynamic
    # Rule 1: Bold + short → Heading 1/2
    if is_bold and word_count <= 12 and not text.endswith('.'):
        level = "Heading 1" if avg_size >= 14 else "Heading 2"
        return True, level

    # Rule 2: Large font + short → Heading 1
    if avg_size >= 14 and word_count <= 12 and not text.endswith('.'):
        return True, "Heading 1"

    # Rule 3: ALL CAPS + short → Heading 1
    if is_all_caps and word_count <= 10:
        return True, "Heading 1"

    # Rule 4: Numbered heading pattern "1. Title" or "1.1 Title"
    if re.match(r'^\d+(\.\d+)*\.?\s+[A-Z]', text) and word_count <= 12:
        dots  = text.split()[0].count('.')
        level = f"Heading {min(dots + 1, 6)}"
        return True, level

    return False, ""


def _is_substantial_cell(text: str) -> bool:
    """Filter table cells: keep if 5+ words and >50% alphabetic (exclude labels, IDs)."""
    text = text.strip()

    if not text:
        return False

    word_count = len(text.split())

    # Too short — likely a label or ID
    if word_count < 5:
        return False

    # Mostly numbers / symbols — table data, not prose
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.5:
        return False

    # Technical labels — UC-1, REQ-1, PER-1 etc
    if re.match(r'^[A-Z]{1,5}-\d+', text):
        return False

    return True


def _extract_table_paragraphs(table) -> list[str]:
    """Extract substantial text from all table cells (row-by-row, cell-by-cell)."""
    paragraphs = []

    for row in table.rows:
        for cell in row.cells:
            # Cell can have multiple paragraphs; join them with space
            cell_text = " ".join(
                p.text.strip()
                for p in cell.paragraphs
                if p.text.strip()
            )

            if _is_substantial_cell(cell_text):
                paragraphs.append(cell_text)

    return paragraphs


def extract_structure(file_bytes: bytes, include_tables: bool = True) -> list[SectionData]:
    """Parse DOCX bytes into structured sections.

    Args:
        include_tables: Grammar=True, Coherence=False
    """
    try:
        doc = Document(io.BytesIO(file_bytes))
    except PackageNotFoundError:
        raise ValueError("Invalid DOCX file.")
    except Exception as e:
        raise ValueError(f"Could not parse DOCX: {e}")

    sections: list[SectionData] = []
    current_section: SectionData | None = None

    # Process doc.element.body to maintain order and include both paragraphs and tables
    for element in doc.element.body:

        # Extract paragraph
        if element.tag == qn('w:p'):
            # python-docx Paragraph object banao
            para = DocxParagraph(element, doc)

            style_name  = para.style.name if para.style else ""
            style_lower = style_name.strip().lower()
            text        = para.text.strip()

            if not text:
                continue

            if style_lower in SKIP_STYLES_LOWER:
                continue

            is_proper_heading = (
                style_lower.startswith("heading") or
                style_lower in HEADING_STYLES_LOWER
            )

            if is_proper_heading:
                if is_skip_heading(text):
                    current_section = None
                    continue
                if current_section:
                    sections.append(current_section)
                current_section = {
                    "heading":       text,
                    "heading_level": get_heading_level(style_name),
                    "paragraphs":    []
                }
                continue

            # Fallback heading detection
            fallback, level = is_fallback_heading(para)
            if fallback:
                if is_skip_heading(text):
                    current_section = None
                    continue
                if current_section:
                    sections.append(current_section)
                current_section = {
                    "heading":       text,
                    "heading_level": level,
                    "paragraphs":    []
                }
                continue

            # Regular paragraph
            if current_section is None:
                current_section = {
                    "heading":       "Document",
                    "heading_level": "Heading 1",
                    "paragraphs":    []
                }

            clean = [p.strip() for p in text.split("\n") if p.strip()]
            current_section["paragraphs"].extend(clean)

        # Extract table
        elif element.tag == qn('w:tbl') and include_tables:  # ← KEY CHANGE
            table = DocxTable(element, doc)

            # Add table content to current section
            if current_section is None:
                current_section = {
                    "heading":       "Document",
                    "heading_level": "Heading 1",
                    "paragraphs":    []
                }

            table_paragraphs = _extract_table_paragraphs(table)
            current_section["paragraphs"].extend(table_paragraphs)

    # Last section
    if current_section:
        sections.append(current_section)

    return sections if sections else [{"heading": "Document", "heading_level": "Heading 1", "paragraphs": []}]



async def extract_document_structure(
    file: UploadFile,
    include_tables: bool = True  
) -> list[SectionData]:
    # FastAPI async wrapper: validate + extract structure.

    filename     = file.filename or ""
    content_type = file.content_type or ""
    content      = await file.read()
    
    # Validate file
    is_valid, error_msg = validate_docx_file(filename, content_type, content)
    if not is_valid:
        raise ValueError(error_msg or "Invalid file")

    return extract_structure(content, include_tables=include_tables)