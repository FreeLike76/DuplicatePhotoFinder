import io
import uuid

from PIL import Image
from fastapi.testclient import TestClient

from main import app

# Utility functions
################################

def create_image_file(
    name: str,
    height: int = 512,
    width: int = 512,
    color: str = "white",
    format: str = "PNG"
) -> Image:
    image = Image.new("RGB", (width, height), color=color)
    
    file = io.BytesIO()
    image.save(file, format)
    
    file.name = f"{name}.{format.lower()}"
    file.seek(0)
    
    return file

def create_blob(size: int) -> bytes:
    file = io.BytesIO(b"\x00" * size)
    file.name = f"blob_{size}.png"
    file.seek(0)
    return file

# Tests
################################

# Had to use in context to trigger model initialization in lifespan
# client = TestClient(app)

def test_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200, "Root endpoint is not available!"

def test_invalid_image_upload():
    with TestClient(app) as client:
        response = client.post("/images")
        assert response.status_code == 422, "No images uploaded, but request was successful."
    
def test_image_upload():
    image_files = [
            ("files", create_image_file("image_white")),                                # OK
            ("files", create_image_file("image_white", format="jpeg")),                 # OK
            ("files", create_image_file("image_red", format="jpeg", color="red")),      # OK
            ("files", create_image_file("image_green", format="tiff", color="green")),  # TIFF
            ("files", create_image_file("image_green", format="tiff", color="green")),  # TIFF
            ("files", create_image_file("image_green", color="green")),                 # OK
            ("files", create_blob(15 * 1024 * 1024))                                    # Too large
        ]
    
    with TestClient(app) as client:
        
        response = client.post("/images", files=image_files)
        assert response.status_code == 200
        body: dict = response.json()

        assert body.get("request_id", None) is not None, "Image upload failed! request_id is missing."
        assert body.get("image_ids", []) == [0, 1, 2, None, None, 3, None], "Images uploaded, but image_ids are incorrect."

def test_duplicate_search():
    image_files = [
        ("files", open("data/test/0.jpg", "rb")),
        ("files", open("data/test/1.jpg", "rb")),
        ("files", open("data/test/2.jpg", "rb")),
        ("files", open("data/test/3.jpg", "rb")),
        ("files", open("data/test/4.png", "rb")),
        ("files", open("data/test/5.png", "rb")),
        ("files", open("data/test/6.png", "rb")),
        ("files", open("data/test/7.png", "rb")),
    ]
    
    with TestClient(app) as client:
        # Upload
        response = client.post("/images", files=image_files)
        assert response.status_code == 200, "Failed to upload images."
        
        # Get request_id
        body: dict = response.json()
        request_id = body.get("request_id", None)
        assert request_id is not None, "Failed to get request_id."
        
        # Find duplicates
        response = client.get(f"/duplicates/{request_id}")
        body: dict = response.json()
        pairs = [(pair["first_image_id"], pair["second_image_id"]) for pair in body]
        
        assert len(pairs) == 2, "Expected 2 pairs of duplicates."
        assert (0, 1) in pairs, "Expected pair (0, 1) to be a duplicate."
        assert (4, 5) in pairs, "Expected pair (4, 5) to be a duplicate."

def test_invalid_duplicate_search():
    with TestClient(app) as client:
        response = client.get(f"/duplicates/{uuid.uuid4()}")
        assert response.status_code == 404
