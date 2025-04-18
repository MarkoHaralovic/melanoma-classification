import torch.nn as nn
from .backbones.ConvNeXt import create_convnext_model 
from .backbones.ConvNeXtV2 import create_convnext_v2_model
from .backbones.EfficientNet import crete_efficientnet_v2_model
from .backbones.DinoV2 import create_dinov2_model

class MelanomaClassifier(nn.Module):
    def __init__(self, model_name='convnext_tiny', num_classes=2, pretrained=True, in_22k=False):
        """
        Initialize the Melanoma Classification model
        
        Args:
            model_name: Name of the ConvNeXt model variant to use
            num_classes: Number of output classes (2 for binary melanoma classification)
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        if model_name.__contains__('convnext_'):
            self.model, self.num_features = create_convnext_model(model_name=model_name, pretrained=pretrained, in22k=in_22k)
            self.model.head = nn.Linear(self.num_features, num_classes)  
        elif model_name.__contains__('efficientnet'):
            self.model, self.num_features  = crete_efficientnet_v2_model(model_name=model_name, num_classes=num_classes, pretrained=pretrained, in22k=in_22k)
        elif model_name.__contains__('convnextv2'):
            self.model, self.num_features = create_convnext_v2_model(model_name=model_name, num_classes = num_classes, pretrained=pretrained, in22k=in_22k)
        elif model_name.__contains__('dinov2'):
            self.model , self.num_features = create_dinov2_model(model_name=model_name, pretrained=pretrained, use_registers = True, freeze=True, register_buffer=None)
            self.model.head.fc = nn.Linear(self.num_features, num_classes, bias=True)  
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
     
    def forward(self, x):
        return self.model(x)