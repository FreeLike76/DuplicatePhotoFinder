from pathlib import Path
from typing import List, Dict
from uuid import UUID

import numpy as np
from PIL import Image

# Local
from .entities import DuplicateImagePair
from .storage import LocalIndexManager
from .models import FeatureExtractor, ModelRegistry

class DuplicatePhotoFinder:
    def __init__(
        self,
        local_storage_dir = Path("data/index"),
        feature_extractor = ModelRegistry.create_swin_v2_s(device="cpu")
    ) -> None:
        self.feature_extractor = feature_extractor
        self.index_manager = LocalIndexManager(
            local_storage_dir,
            self.feature_extractor.features_size
        )

    def shutdown(self):
        # TODO: Gracefull shutdown for DB & other services
        # Not needed rn
        pass

    def create_collection(
        self,
        images: List[Image.Image]
    ) -> UUID:
        # Create & fill index with image features
        vector_index = self.index_manager.create_index()
        for i, image in enumerate(images):
            features = self.feature_extractor.inference(image)
            vector_index.add_item(i, features)
        
        # TODO: tune n_trees
        n_trees = len(images) // 5 + 1
        n_trees = min(25, max(10, n_trees))
        
        # Save under unique id
        vector_index.build(n_trees)
        collection_id = self.index_manager.save_index(
            vector_index,
            unload=True
        )
        
        return collection_id
    
    def find_duplicates(
        self,
        collection_id: UUID,
        threshold: float = 0.75
    ) -> List[DuplicateImagePair]:
        # Load index
        vector_index = self.index_manager.load_index(collection_id)
        
        # Pairwise comparison
        search_results = []
        dot_threshold = threshold * 2 - 1 # [0, 1] -> [-1, 1]
        
        n_items = vector_index.get_n_items()
        for i in range(n_items):
            for j in range(i + 1, n_items):
                # Compare
                dot_product = vector_index.get_distance(i, j)
                if dot_product < dot_threshold:
                    continue
                
                # Save
                search_match = DuplicateImagePair(
                    first_image_id=i,
                    second_image_id=j,
                    similarity=(dot_product + 1) / 2 # [-1, 1] -> [0, 1]
                )
                search_results.append(search_match)
        
        vector_index.unload()
        return search_results