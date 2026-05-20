import io
import logging
import re
import uuid

from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph as DocxParagraph
from docx.table import Table as DocxTable
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
)

from app.core import settings, client, init_collection
from app.core.models import ChunkData
from app.core.embedding import get_embedding_model

logger = logging.getLogger(__name__)


def extract_text_from_docx(file_bytes: bytes) -> list[dict]:
    """Extract paragraphs and table cells from DOCX file."""
    doc = Document(io.BytesIO(file_bytes))
    sections_data: list[dict] = []

    # Iterate through all body elements maintaining document order
    for element in doc.element.body:
        # Extract paragraph text
        if element.tag == qn('w:p'):
            para = DocxParagraph(element, doc)
            text = " ".join((para.text or "").split())
            if text:
                sections_data.append({"text": text})
        
        # Extract table content (aggregating by row rather than individual cells)
        elif element.tag == qn('w:tbl'):
            table = DocxTable(element, doc)
            for row in table.rows:
                row_cells_text = []
                for cell in row.cells:
                    cell_text = " ".join(
                        p.text.strip()
                        for p in cell.paragraphs
                        if p.text.strip()
                    )
                    if cell_text:
                        row_cells_text.append(cell_text)
                
                # If row has content, add it as a single chunk to preserve relational context
                if row_cells_text:
                    sections_data.append({"text": " | ".join(row_cells_text)})

    return sections_data


def chunk_content(sections_data: list[dict], document_id: str) -> list[ChunkData]:
    """Split content into chunks maintaining sentence boundaries with overlap."""
    all_chunks: list[ChunkData] = []

    chunk_size = int(settings.chunk_size)
    chunk_overlap = int(settings.chunk_overlap)

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")

    for section in sections_data:
        # Split by sentence boundaries (., !, ?)
        sentences = re.split(r"(?<=[.!?])\s+", section["text"])
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            continue

        current_chunk: list[str] = []
        current_size = 0

        for sentence in sentences:
            sentence_len = len(sentence.split())

            # Check if sentence exceeds chunk size limit
            if current_size + sentence_len > chunk_size:
                # Save current chunk
                if current_chunk:
                    all_chunks.append(
                        {
                            "text": " ".join(current_chunk),
                            "document_id": document_id,
                        }
                    )

                # Apply overlap: reuse last N words from previous chunk
                if chunk_overlap > 0 and len(current_chunk) > 0:
                    prev_chunk_text = " ".join(current_chunk)
                    prev_words = prev_chunk_text.split()
                    overlap_words = prev_words[-chunk_overlap:]
                    current_chunk = [" ".join(overlap_words), sentence]
                else:
                    current_chunk = [sentence]
                
                # Recalculate chunk size
                current_size = len(" ".join(current_chunk).split())
            else:
                # Sentence fits, add to current chunk
                current_chunk.append(sentence)
                current_size += sentence_len

        # Save final chunk
        if current_chunk:
            all_chunks.append(
                {
                    "text": " ".join(current_chunk),
                    "document_id": document_id,
                }
            )

    return all_chunks


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Generate embeddings for text chunks using FastEmbed model."""
    if not chunks:
        raise ValueError("No chunks provided for embedding")

    embedding_model = get_embedding_model()
    vectors = [v.tolist() for v in embedding_model.embed(chunks)]
    if not vectors:
        raise ValueError("Embedding model returned no vectors")

    return vectors


def _extract_collection_vector_size() -> int:
    """Get embedding vector dimension from Qdrant collection."""
    try:
        info = client.get_collection(settings.qdrant_collection_name)
    except UnexpectedResponse as exc:
        if "not found: collection" in str(exc).lower() or "doesn't exist" in str(exc).lower():
            init_collection()
            info = client.get_collection(settings.qdrant_collection_name)
        else:
            raise

    vectors_config = info.config.params.vectors

    # Single unnamed vector
    if hasattr(vectors_config, "size"):
        return int(vectors_config.size)

    # Named vectors
    if isinstance(vectors_config, dict) and vectors_config:
        first_cfg = next(iter(vectors_config.values()))
        if hasattr(first_cfg, "size"):
            return int(first_cfg.size)

    raise ValueError("Could not detect collection vector size")


def _delete_document_points(document_id: str, exclude_batch_id: str | None = None) -> None:
    """Remove previous indexed chunks for a document (keeps current batch only)."""
    must_not = []
    if exclude_batch_id:
        must_not.append(
            FieldCondition(
                key="batch_id",
                match=MatchValue(value=exclude_batch_id),
            )
        )

    selector = FilterSelector(
        filter=Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ],
            must_not=must_not,
        )
    )
    # Delete old points, keeping only latest batch
    client.delete(
        collection_name=settings.qdrant_collection_name,
        points_selector=selector,
        wait=True,
    )


def store_chunks(
    chunks_data: list[ChunkData],
    embeddings: list[list[float]],
    document_id: str,
) -> None:
    """Store chunks with embeddings in Qdrant, replacing old versions of document."""
    if len(chunks_data) != len(embeddings):
        raise ValueError("Chunks and embeddings count mismatch")

    if not chunks_data:
        raise ValueError("No chunk data to store")

    if not embeddings or not embeddings[0]:
        raise ValueError("Invalid embeddings generated")

    # Validate embedding dimensions match collection
    expected_dim = _extract_collection_vector_size()
    actual_dim = len(embeddings[0])
    if actual_dim != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: model produced {actual_dim}, collection expects {expected_dim}"
        )

    # Create unique batch ID for this indexing operation
    batch_id = uuid.uuid4().hex

    # Build point objects with vectors and metadata
    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{batch_id}:{i}")),
            vector=embedding,
            payload={
                "text": chunk["text"],
                "document_id": chunk["document_id"],
                "batch_id": batch_id,
            },
        )
        for i, (chunk, embedding) in enumerate(zip(chunks_data, embeddings))
    ]

    try:
        # Store points in Qdrant
        client.upsert(
            collection_name=settings.qdrant_collection_name,
            points=points,
            wait=True,
        )
    except UnexpectedResponse as exc:
        if "not found: collection" in str(exc).lower() or "doesn't exist" in str(exc).lower():
            init_collection()
            client.upsert(
                collection_name=settings.qdrant_collection_name,
                points=points,
                wait=True,
            )
        else:
            raise

    # Remove previous versions of this document
    _delete_document_points(document_id, exclude_batch_id=batch_id)