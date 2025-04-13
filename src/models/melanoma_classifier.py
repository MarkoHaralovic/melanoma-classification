import torch.nn as nn
from .backbones.ConvNeXt import convnext_tiny, convnext_small, convnext_base, convnext_large
from .backbones.EfficientNet import crete_efficientnet_v2_model

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
        
        if model_name == 'convnext_tiny':
            self.model = convnext_tiny(pretrained=pretrained, in_22k=in_22k)
            self.model.head = nn.Linear(768, num_classes)
                
        elif model_name == 'convnext_small':
            self.model = convnext_small(pretrained=pretrained, in_22k=in_22k)
            self.model.head  = nn.Linear(768, num_classes)
                
        elif model_name == 'convnext_base':
            self.model = convnext_base(pretrained=pretrained, in_22k=in_22k)
            self.model.head  = nn.Linear(1024, num_classes)
                
        elif model_name == 'convnext_large':
            self.model = convnext_large(pretrained=pretrained, in_22k=in_22k)
            self.model.head  = nn.Linear(1536, num_classes)
        elif model_name == 'convnext_xlarge':
            self.model = convnext_large(pretrained=pretrained, in_22k=True)
            self.model.head  = nn.Linear(2048, num_classes)
        elif model_name.__contains__('efficientnet'):
            self.model, num_features  = crete_efficientnet_v2_model(model_name=model_name, num_classes=num_classes, pretrained=pretrained, in22k=in_22k)
            self.model.classifier = nn.Linear(num_features, num_classes)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
            
    def forward(self, x):
        return self.model(x)