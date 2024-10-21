from uuid import UUID
from fastapi import FastAPI, Request, Response, HTTPException
from contextlib import asynccontextmanager

from src.service import DuplicatePhotoFinder

# Init
################################

app: FastAPI = None
dpf_service: DuplicatePhotoFinder = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global dpf_service
    dpf_service = DuplicatePhotoFinder()
    yield
    dpf_service.shutdown()

app = FastAPI(lifespan=lifespan)

# Endpoints
################################

@app.get("/")
def get_root():
    return {
        "name": "DuplicatePhotoFinder",
        "Status": "OK"
    }

@app.post("/images")
def post_images(request: Request):
    """
    Get multiple images as multipart/form-data
    Compute embeddings for each image
    Store embeddings in qdrant
    Return request_id & list of statuses for each image
    """
    pass

@app.get("/duplicates/{request_id}")
def get_duplicates(request_id: UUID, threshold: float = 0.9):
    """
    Find duplicates for a list of images with the given request_id
    Return list of duplicates
    """
    