from typing import List
from pathlib import Path
from uuid import UUID, uuid4

from PIL import Image
from annoy import AnnoyIndex

# Local
from .models import FeatureExtractor, create_efficientnet_v2_s

import numpy as np

class DuplicatePhotoFinder:
    def __init__(
        self,
        feature_extractor: FeatureExtractor = create_efficientnet_v2_s(device="cuda")
    ) -> None:
        self.feature_extractor = feature_extractor

    def shutdown(self):
        # TODO: gracefull shutdown
        pass

    def create_collection(
        self,
        images: List[Image.Image]
    ) -> UUID:
        # TODO: unique id
        collection_id = uuid4()
        vector_storage = AnnoyIndex(self.feature_extractor.features_size, "dot")
        
        for i, image in enumerate(images):
            features = self.feature_extractor.inference(image)
            vector_storage.add_item(i, features)
        
        # TODO: tree size
        vector_storage.build(len(images) // 4 + 1)
        vector_storage.save(f"data/{str(collection_id)}.ann")
        vector_storage.unload()
        
        return collection_id
    
    def find_duplicates(
        self,
        collection_id: UUID,
        threshold: float = 0.9
    ) -> List[List[int]]:
        # Load
        index_p = Path(f"data/{str(collection_id)}.ann")
        if not index_p.exists() or not index_p.is_file():
            raise FileNotFoundError(f"Index {index_p} not found.")
        
        vector_storage = AnnoyIndex(self.feature_extractor.features_size, "dot")
        vector_storage.load(f"data/{str(collection_id)}.ann")
        
        # Find duplicates
        #n_items = vector_storage.get_n_items()
        #n_neighbours = int(n_items // 4 + 1)
        #search_results = [[] for _ in range(n_items)]
        
        #for i in range(n_items):
        #    neighbours, distances = vector_storage.get_nns_by_item(
        #        i, n_neighbours,
        #        include_distances=True
        #    )
        #    
        #    # TODO: Threshold
        #    for j, dist in zip(neighbours, distances):
        #        #if dist < threshold:
        #        duplicate = {
        #            "index": i,
        #            "neighbour": j,
        #            "distance": dist
        #        }
        #        search_results[i].append(duplicate)
        
        # Find duplicates
        n_items = vector_storage.get_n_items()
        #search_results = [[] for _ in range(n_items)]
        dist_matrix = np.zeros((n_items, n_items), dtype=np.float32)
        for i in range(n_items):
            for j in range(i + 1, n_items):
                dist = vector_storage.get_distance(i, j)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        vector_storage.unload()
        
        search_results = dist_matrix.tolist()
        return search_results