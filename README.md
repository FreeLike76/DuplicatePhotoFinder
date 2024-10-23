# Photo Duplicate Finder

This service provides an API for detecting duplicate photos using deep learning models. It allows users to upload multiple images, stores them as feature vectors and identifies duplicates based on features similarity.

## Technologies Used

- **FastAPI**: a modern web framework for building APIs with Python.
- **Uvicorn**: ASGI server implementation, used to run the FastAPI application.
- **Pydantic**: for data validation and management.
- **PyTorch & TorchVision**: for image feature extraction using pretrained models.
- **Annoy**: library for vector storage and approximate nearest neighbor searches.
- **Pillow**: image processing library.

## Available Models (ImageNet Pretrained)

| Model Name | Top-1 Accuracy | Top-5 Accuracy | Parameters | FLOPS |
|------------|----------------|----------------|------------|-------|
| EfficientNet V2 Small | 84.228 | 96.878 | 21.5M | 8.37 |
| MaxVit Tiny | 83.7 | 96.722 | 30.9M | 5.56 |
| **Swin V2 Small** | 83.712 | 96.816 | 49.7M | 11.55 |

## How to Run

The service is containerized using Docker, making it easy to deploy and run anywhere.

### Docker

1. Make sure you have Docker installed on your machine.
2. Clone the repository.
    ```bash
    git clone https://github.com/FreeLike76/LearnOpenGL.git
    cd DuplicatePhotoFinder
    ```
3. Build and start the service
    ```bash
    docker-compose up
    ```
    This will build the Docker image and start the service on port 8000.

### Manually

Alternatively, you can run the service manually on your local machine.

1. Clone the repository.
    ```bash
    git clone URL
    cd DuplicatePhotoFinder
    ```
2. Create `Python 3.12` environment using `venv` or `conda`.
3. Clone the repository.
    ```bash
    git clone https://github.com/FreeLike76/LearnOpenGL.git
    cd DuplicatePhotoFinder
    ```
4. Install dependencies.
    ```bash
    pip install -r requirements.txt
    ```
5. Start the service using Uvicorn.
    ```bash
    uvicorn main:app --host IP --port PORT
    ```

### Testing

To verify that everything is working correctly, a test.py file is provided. This file includes various tests to ensure the functionality of the service. A python environment is required.
```bash
python test.py
```

## Documentation

### Code

- **DuplicatePhotoFinderService**: Main service class that handles the core functionality.

- **FeatureExtractor**: Encapsulates data preprocessing, model inference and feature postprocessing.

- **LocalIndexManager**: Manages vector index creation using Annoy, handles local index files (save/load).

### API

- **`GET` /**: Root endpoint to ensure the server is running & healhty.

- **`POST` /images**: Upload a collection of images. Returns a unique request_id: UUID, used for future requests, and an indexed list of successfully uploaded images (or None if an image failed to meet the requirements).

- **`GET` /duplicates/{*request_id*}**: Search for duplicate images within a specified collection. A path parameter request_id: UUID must be provided. Additionally, you can include an optional threshold: float (0.75) query parameter to specify the minimum similarity score. Returns a list of duplicate image pairs along with their similarity scores.

## Future plans:

- [ ] Add logger for better debugging.
- [ ] Add support for loading configuration files.
- [ ] CUDA support for faster inference (docker container).
- [ ] Async/batch image processing.
- [ ] Support for additional DL models.
- [ ] Alternative vector storage solutions.