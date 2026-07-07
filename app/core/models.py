# app/core/models.py
"""Shared data models and types."""

from typing import TypedDict


class SectionData(TypedDict):
    """Represents a document section with heading and paragraphs."""
    heading: str
    heading_level: str  # "Heading 1", "Heading 2", etc
    paragraphs: list[str]


class SentenceIssue(TypedDict):
    """Represents a coherence issue between sentences."""
    heading: str
    level: str
    location: str
    score: float
    sentence_1: str
    sentence_2: str


class ParagraphIssue(TypedDict):
    """Represents a coherence issue between paragraphs."""
    heading: str
    level: str
    location: str
    score: float
    paragraph_1: str
    paragraph_2: str


class ChunkData(TypedDict):
    """Represents a document chunk for vector storage."""
    text: str
    document_id: str
