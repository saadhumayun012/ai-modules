from fastapi import APIRouter
from .chat_query import router as chat_query__router
from .chat_indexing import router as chat_indexing_router
from .coherence import router as coherence_router
from .grammar import router as grammar_router

router = APIRouter()

router.include_router(chat_query__router)
router.include_router(chat_indexing_router)
router.include_router(coherence_router)
router.include_router(grammar_router)