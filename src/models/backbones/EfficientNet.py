""" 
   EfficientNet model for image classification. Using timm library for model definition.
"""

import timm

def crete_efficientnet_v2_model(model_name='efficientnetv2_m', num_classes=2, pretrained=True, in_22k=False):
   """
   Create an EfficientNet model for image classification.

   Args:
      model_name (str): Name of the EfficientNet model variant to use.
      num_classes (int): Number of output classes (e.g. 0 for not initializing head).
      pretrained (bool): Whether to use pretrained weights.

   Returns:
      model: The EfficientNet model.
   """

   if not model_name.startswith('tf_'):
      model_name = 'tf_' + model_name
      
   model_name += '.in21k' if in_22k else '.in21k_ft_in1k'

   print(f"Creating EfficientNet model: {model_name}")
   model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
   num_features = model.classifier.in_features

   return model, num_features