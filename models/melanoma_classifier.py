import torch
import torch.nn as nn
from .convnext import convnext_tiny, convnext_small, convnext_base, convnext_large

class MelanomaClassifier(nn.Module):
    def __init__(self, model_name='convnext_tiny', num_classes=2, pretrained=True):
        """
        Initialize the Melanoma Classification model
        
        Args:
            model_name: Name of the ConvNeXt model variant to use
            num_classes: Number of output classes (2 for binary melanoma classification)
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        if model_name == 'convnext_tiny':
            self.backbone = convnext_tiny(pretrained=pretrained)
            if pretrained:
                self.backbone.head.fc = nn.Linear(768, num_classes)
                
        elif model_name == 'convnext_small':
            self.backbone = convnext_small(pretrained=pretrained)
            if pretrained:
                self.backbone.head.fc = nn.Linear(768, num_classes)
                
        elif model_name == 'convnext_base':
            self.backbone = convnext_base(pretrained=pretrained)
            if pretrained:
                self.backbone.head.fc = nn.Linear(1024, num_classes)
                
        elif model_name == 'convnext_large':
            self.backbone = convnext_large(pretrained=pretrained)
            if pretrained:
                self.backbone.head.fc = nn.Linear(1536, num_classes)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
            
    def forward(self, x):
        return self.backbone(x)