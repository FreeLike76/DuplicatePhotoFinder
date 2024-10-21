from typing import Callable

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
#import torch.nn.functional as F
from torchvision import models, transforms

class FeatureExtractor:
    def __init__(
        self,
        model: nn.Module,
        transforms: Callable[[Image.Image], torch.Tensor],
        device: torch.device = torch.device("cpu")
    ) -> None:
        self.device = device
        self.transforms = transforms
        self.model = model
        
        self.model.to(self.device)
        self.model.eval()
        # FIXME: Not working
        #self.model.compile()
    
    def inference(
        self,
        x: Image.Image
    ) -> np.ndarray:
        #with torch.no_grad():
        with torch.inference_mode():    
            image_pt = self.transforms(x)
            image_pt = image_pt.unsqueeze(0)
            image_pt = image_pt.to(self.device)
            
            out_pt: torch.Tensor = self.model(image_pt)
            out_np = out_pt.cpu().numpy()
        
        return out_np

class EfficientNetV2S(FeatureExtractor):
    def __init__(
        self,
        device: torch.device = torch.device("cpu")
    ) -> None:
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        transforms = weights.transforms()
        
        model = models.efficientnet_v2_s(weights=weights)
        model.classifier = nn.Identity()
        
        FeatureExtractor.__init__(self, model, transforms, device)

class SwinV2S(FeatureExtractor):
    def __init__(
        self,
        device: torch.device = torch.device("cpu")
    ) -> None:
        weights = models.Swin_V2_S_Weights.IMAGENET1K_V1
        transforms = weights.transforms()
        
        model = models.swin_v2_s(weights=weights)
        model.head = nn.Identity()
        
        FeatureExtractor.__init__(self, model, transforms, device)