from typing import List, Callable

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models

class FeatureExtractor:
    def __init__(
        self,
        features_size: int,
        model: nn.Module,
        transforms: Callable[[Image.Image], torch.Tensor],
        normalize_features: bool = True,
        device: torch.device = torch.device("cpu"),
        verbose: bool = True
    ) -> None:
        self.features_size = features_size
        self.transforms = transforms
        self.model = model
        
        self.normalize_features = normalize_features
        self.device = device
        self.verbose = verbose
        
        self.model.to(self.device)
        self.model.eval()
        # FIXME: Driver error
        # self.model.compile()
    
    def inference(
        self,
        x: Image.Image
    ) -> np.ndarray:
        with torch.inference_mode():    
            image_pt = self.transforms(x)
            image_pt = image_pt.unsqueeze(0)
            image_pt = image_pt.to(self.device)
            
            out_pt: torch.Tensor = self.model(image_pt)
            out_np = out_pt.squeeze(0).cpu().numpy()
        
        if self.normalize_features:
            out_np = self.normalize(out_np)
        
        return out_np

    def normalize(
        self,
        x: np.ndarray,
        eps: float = 1e-6
    ) -> np.ndarray:
        norm = np.linalg.norm(x)
        if norm < eps:
            if self.verbose: print("Warning! Encountered zero norm vector.")
            return np.zeros_like(x)
        x_norm = x / norm
        return x_norm
    
    #def inference_batch(
    #    self,
    #    x: List[Image.Image]
    #) -> np.ndarray:
    #    with torch.inference_mode():
    #        images_pt = torch.stack([self.transforms(image) for image in x])
    #        images_pt = images_pt.to(self.device)
    #        
    #        out_pt: torch.Tensor = self.model(images_pt)
    #        out_np = out_pt.cpu().numpy()
    #    
    #    return out_np

class ModelRegistry:
    
    @staticmethod
    def create_efficientnet_v2_s(**kwargs) -> FeatureExtractor:
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        transforms = weights.transforms()

        model = models.efficientnet_v2_s(weights=weights)
        model.classifier = nn.Identity()

        extractor = FeatureExtractor(1280, model, transforms, **kwargs)
        return extractor

    @staticmethod
    def create_swin_v2_s(**kwargs) -> FeatureExtractor:
        weights = models.Swin_V2_S_Weights.IMAGENET1K_V1
        transforms = weights.transforms()

        model = models.swin_v2_s(weights=weights)
        model.head = nn.Identity()

        extractor = FeatureExtractor(768, model, transforms, **kwargs)
        return extractor

    @staticmethod
    def create_max_vit_tiny(**kwargs) -> FeatureExtractor:
        weights = models.MaxVit_T_Weights.IMAGENET1K_V1
        transforms = weights.transforms()

        model = models.maxvit_t(weights=weights)
        model.classifier[-1] = nn.Identity()
        
        extractor = FeatureExtractor(512, model, transforms, **kwargs)
        return extractor