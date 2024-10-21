from typing import List
from uuid import UUID, uuid4

from PIL import Image

# Local
from models import EfficientNetV2S

class DuplicatePhotoFinder:
    def __init__(
        self
    ) -> None:
        # TODO: Load torch model & qdrant client
        self.feature_extractor = EfficientNetV2S()
        self.db_client = None

    def shutdown(self):
        # TODO: gracefull shutdown
        pass

    def create_collection(
        self,
        images: List[Image.Image]
    ) -> UUID:
        # db, get collection_id
        # for each image in images:
        #     embedding = self.feature_extractor.inference(image)
        #     db, insert embedding
        # db, save
        # return collection_id
        pass