from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Setting(BaseSettings):
    base_url: str
    api_key: str
    qdrant_url: str
    qdrant_collection_name: str
    embedding_model: str
    llm_model: str
    grammar_api_url: str

    chunk_size: int = 200
    chunk_overlap: int = 20
    retrieval_top_k: int = 10
    retrieval_score_threshold: float = 0.15
    indexing_max_docx_bytes: int = 10 * 1024 * 1024

    coherence_sentence_threshold: float = 0.5
    coherence_paragraph_threshold: float = 0.4
    coherence_sentence_window: int = 2
    coherence_min_sentence_words: int = 5
    coherence_max_docx_bytes: int = 10 * 1024 * 1024
    
    # Grammar API settings
    grammar_api_timeout: float = 60.0  # seconds
    
    # File upload limits (unified)
    file_max_bytes: int = 10 * 1024 * 1024  # 10 MB
    
    # LLM chat settings
    llm_temperature: float = 0.3
    llm_max_tokens: int = 800 
    
    # Retrieval quality settings
    retrieval_enforce_threshold: bool = True  # If True, reject results below threshold instead of falling back
    retrieval_deduplication_enabled: bool = True
    retrieval_deduplication_threshold: float = 0.75  # Similarity score (0-1) for duplicate detection
    retrieval_max_context_length: Optional[int] = None  # Max characters for context (None = unlimited)
    
    # Query expansion settings
    query_expansion_enabled: bool = True
    query_expansion_use_similarity: bool = True  # Use text similarity in re-ranking
    
    # Response cleaning settings
    response_preserve_formatting: bool = False  # If True, preserves markdown formatting
    
    model_config = SettingsConfigDict(env_file=".env")

settings = Setting()