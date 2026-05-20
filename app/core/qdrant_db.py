from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.core import settings
from app.core.embedding import get_embedding_model

client = QdrantClient(url=settings.qdrant_url)


def _resolve_embedding_dimension() -> int:
    # Get embedding vector size by running inference on test input
    model = get_embedding_model()
    vector = next(model.embed(["dimension probe"]))
    return int(len(vector))


def _get_existing_collection_dimension() -> int:
    # Extract vector dimension from existing Qdrant collection config
    info = client.get_collection(settings.qdrant_collection_name)
    vectors_config = info.config.params.vectors

    if hasattr(vectors_config, "size"):
        return int(vectors_config.size)

    if isinstance(vectors_config, dict) and vectors_config:
        first_cfg = next(iter(vectors_config.values()))
        if hasattr(first_cfg, "size"):
            return int(first_cfg.size)

    raise ValueError("Could not detect existing collection vector size")

def init_collection() -> None:
    # Create collection if missing or validate existing collection's vector size
    exists = client.collection_exists(settings.qdrant_collection_name)
    expected_dim = _resolve_embedding_dimension()
    
    if not exists:
        client.create_collection(
            collection_name= settings.qdrant_collection_name,
            vectors_config= VectorParams(
                size=expected_dim,
                distance=Distance.COSINE
            ) 
        )
        return

    existing_dim = _get_existing_collection_dimension()
    if existing_dim != expected_dim:
        raise ValueError(
            f"Collection vector size mismatch: existing={existing_dim}, expected={expected_dim}. "
            "Recreate the collection or align embedding_model."
        )

def clear_collection() -> None:
    # Delete collection and recreate (useful for resetting)
    client.delete_collection(settings.qdrant_collection_name)
    init_collection()