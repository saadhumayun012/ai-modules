from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.routes import router
from app.core import init_collection

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Qdrant collection on startup
    init_collection()
    yield


app = FastAPI(lifespan=lifespan)

app.router.include_router(router)