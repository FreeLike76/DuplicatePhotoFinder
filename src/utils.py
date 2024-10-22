from io import BytesIO
from typing import List, Union
from fastapi import UploadFile
from PIL import Image, ImageOps

def is_valid_image_file(
    file: UploadFile,
    supported_formats: List[str] = ["image/png", "image/jpeg"],
    max_size: Union[None, int] = 10 * 1024 * 1024,
    verbose: bool = True
) -> bool:
    # Check type
    if file.content_type not in supported_formats:
        if verbose: print(f"Warning! Unsupported file type {file.content_type}.")
        return False
    
    # Check size
    if max_size is not None and (file.size is None or file.size > max_size):
        if verbose: print(f"Warning! File size {file.size} is missing or too large.")
        return False

    return True

def open_image(data: bytes) -> Image.Image:
    image = Image.open(BytesIO(data))
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image