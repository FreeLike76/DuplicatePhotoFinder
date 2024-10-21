from io import BytesIO
from uuid import UUID
from typing import List
from contextlib import asynccontextmanager

from PIL import Image, ImageOps
from fastapi import FastAPI, UploadFile, File, HTTPException

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

def open_image(data: bytes) -> Image.Image:
    image = Image.open(BytesIO(data))
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

@app.get("/")
def get_root():
    return {
        "name": "DuplicatePhotoFinder",
        "Status": "OK"
    }

@app.post("/images")
async def post_images(files: List[UploadFile] = File(...)):
    images = []
    for i, file in enumerate(files):
        if file.content_type not in ["image/png", "image/jpeg"]:
            print(f"Skipping unsupported file {file.content_type}")
            continue
        
        # TODO: try catch
        contents = await file.read()
        image = open_image(contents)
        images.append(image)
    
    request_id: UUID = dpf_service.create_collection(images)
    response = {
        "request_id": str(request_id),
        "images": len(images)
    }
    return response

@app.get("/duplicates/{request_id}")
def get_duplicates(request_id: UUID, threshold: float = 0.9):
    """
    Find duplicates for a list of images with the given request_id
    Return list of duplicates
    """
    duplicates = dpf_service.find_duplicates(request_id, threshold)
    return duplicates

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)