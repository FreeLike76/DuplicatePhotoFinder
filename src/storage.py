import uuid
from pathlib import Path
from annoy import AnnoyIndex
from typing import Literal

class LocalIndexManager:
    def __init__(
        self,
        local_dir: Path,
        features_size: int,
        metric: Literal["angular", "euclidean", "manhattan", "hamming", "dot"] = "dot"
    ) -> None:
        # Make sure the local_dir is dir
        if not local_dir.is_dir():
            raise ValueError(f"Invalid local_dir: {local_dir}")
        if not local_dir.exists():
            local_dir.mkdir(parents=True, exist_ok=True)
        
        self.local_dir = local_dir
        self.feature_size = features_size
        self.metric = metric
    
    def _format_file_path(self, index_id: uuid.UUID) -> Path:
        return self.local_dir / f"{str(index_id)}.ann"
    
    def create_index(self) -> AnnoyIndex:
        vector_index = AnnoyIndex(self.feature_size, self.metric)
        return vector_index
    
    def save_index(
        self,
        vector_index: AnnoyIndex,
        unload: bool = False
    ) -> uuid.UUID:
        # Generate unique id
        index_id = uuid.uuid4()
        while self._format_file_path(index_id).exists():
            index_id = uuid.uuid4()
        
        # Save
        index_file_p = self._format_file_path(index_id)
        vector_index.save(str(index_file_p))
        if unload: vector_index.unload()
        
        return index_id
    
    def load_index(
        self,
        index_id: uuid.UUID,
        prefault: bool = False
    ) -> AnnoyIndex:
        # Find file
        index_file_p = self._format_file_path(index_id)
        if not index_file_p.exists() or not index_file_p.is_file():
            raise FileNotFoundError(f"Index file ({index_file_p}) not found.")
        
        # Load
        vector_index = self.create_index()
        vector_index.load(str(index_file_p), prefault=prefault)
        
        return vector_index