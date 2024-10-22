import uuid
from typing import List, Union
from pydantic import BaseModel

class ImageUploadResult(BaseModel):
    request_id: uuid.UUID
    image_ids: List[Union[int, None]]

class DuplicateImagePair(BaseModel):
    first_image_id: int
    second_image_id: int
    similarity: float