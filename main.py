from uuid import UUID
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException

from src.service import DuplicatePhotoFinder
from src.entities import ImageUploadResult, DuplicateImagePair
from src.utils import is_valid_image_file, open_image

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
        "name": "Duplicate Photo Finder",
        "Status": "OK"
    }

@app.post("/images")
async def post_images(files: List[UploadFile] = File(...)) -> ImageUploadResult:
    images = []
    image_ids = []
    image_id = 0
    
    for file in files:
        # Validate
        if not is_valid_image_file(file):
            image_ids.append(None)
            continue
        # Open
        try:
            contents = await file.read()
            image = open_image(contents)
            
            images.append(image)
            image_ids.append(image_id)
            image_id += 1
        
        except Exception as e:
            print(f"Image reading error! Message: {e}")
            image_ids.append(None)
    
    if len(images) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least two valid images are required."
        )
    
    # Process    
    request_id: UUID = dpf_service.create_collection(images)
    response = ImageUploadResult(
        request_id=request_id,
        image_ids=image_ids
    )
    return response

@app.get("/duplicates/{request_id}")
def get_duplicates(request_id: UUID, threshold: float = 0.75) -> List[DuplicateImagePair]:
    try:
        duplicates = dpf_service.find_duplicates(request_id, threshold)
    
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Collection not found."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {e}"
        )
    return duplicates

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)