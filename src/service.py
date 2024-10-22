from pathlib import Path
from typing import List
from uuid import UUID

import numpy as np
from PIL import Image

# Local
from .storage import LocalIndexManager
from .models import FeatureExtractor, ModelRegistry

class DuplicatePhotoFinder:
    def __init__(
        self,
        local_storage_dir = Path("data/index"),
        feature_extractor = ModelRegistry.create_swin_v2_s(device="cuda")
    ) -> None:
        self.feature_extractor = feature_extractor
        self.index_manager = LocalIndexManager(
            local_storage_dir,
            self.feature_extractor.features_size
        )

    def shutdown(self):
        # TODO: gracefull shutdown
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
        threshold: float = 0.9
    ) -> List[List[int]]:
        # Load index
        vector_index = self.index_manager.load_index(collection_id)
        dot_threshold = threshold * 2 - 1 # [0, 1] -> [-1, 1]
        
        # Find duplicates
        n_items = vector_index.get_n_items()
        search_results = [[] for _ in range(n_items)]
        
        ## Pairwise
        #dist_matrix = np.zeros((n_items, n_items), dtype=np.float32)
        #for i in range(n_items):
        #    for j in range(i + 1, n_items):
        #        dist = vector_index.get_distance(i, j)
        #        dist_matrix[i, j] = dist
        #        dist_matrix[j, i] = dist
        #search_results = dist_matrix.tolist()
        
        vector_index.unload()
        return search_results