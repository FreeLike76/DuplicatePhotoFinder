from typing import List
from uuid import UUID, uuid4

from PIL import Image

# Local
from .models import FeatureExtractor, EfficientNetV2S

class DuplicatePhotoFinder:
    def __init__(
        self,
        feature_extractor: FeatureExtractor = EfficientNetV2S("cuda")
    ) -> None:
        # TODO: Load torch model & qdrant client
        self.feature_extractor = feature_extractor
        self.db_client = None

    def shutdown(self):
        # TODO: gracefull shutdown
        pass

    def create_collection(
        self,
        images: List[Image.Image]
    ) -> UUID:
        for image in images:
            features = self.feature_extractor.inference(image)
            print(features)
        return uuid4()
    
        # db, get collection_id
        # for each image in images:
        #     embedding = self.feature_extractor.inference(image)
        #     db, insert embedding
        # db, save
        # return collection_id
        pass
    
    def find_duplicates(
        self,
        collection_id: UUID,
        threshold: float = 0.9
    ) -> List[int]:
        # db, try load
        # for each embedding in collection:
        #     db, find similar embeddings
        #    if similarity > threshold:
        #        mark
        # return list of image indexes
        pass