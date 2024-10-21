import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

class EfficientNetV2S(nn.Module):
    def __init__(
        self,
        device: torch.device = torch.device("cpu")
    ) -> None:
        nn.Module.__init__(self)
        
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        self.model = models.efficientnet_v2_s(weights=weights)
        self.model.classifier = nn.Identity()
        self.transform = weights.transforms
        
        self.device = device
        self.transform = self.transform.to(device)
        self.model = self.model.to(device)
    
    def inference(
        self,
        x: Image.Image
    ) -> torch.Tensor:
        
        with torch.no_grad():
            image_pt = self.transform(x)
            image_pt = image_pt.unsqueeze(0)
            image_pt = image_pt.to(self.device)
            
            out_pt = self.model(x)
            out_np = out_pt.cpu().numpy()
        
        return out_np