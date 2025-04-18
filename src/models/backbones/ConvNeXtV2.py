import timm

def create_convnext_v2_model(model_name='convnextv2_atto', num_classes=2, pretrained=True, in22k=False):
   """
   Create a ConvNeXtV2 model for image classification using timm.

   Args:
      model_name (str): Base name of the ConvNeXtV2 variant (e.g., 'convnextv2_atto').
      num_classes (int): Number of output classes (e.g., 2 for binary classification).
      pretrained (bool): Whether to use pretrained weights.
      in22k (bool): Whether to use ImageNet-22k pretraining or fine-tuned on 1k.

   Returns:
      model (nn.Module): The created model.
      num_features (int): Number of features before the classifier.
   """
   if in22k:
      model_name += '.in22k'  
   else:
      model_name += '.fcmae_ft_in1k'  

   print(f"Creating ConvNeXtV2 model: {model_name}")
   
   model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

   if hasattr(model, 'classifier') and hasattr(model.classifier, 'in_features'):
      num_features = model.classifier.in_features
   else:
      num_features = model.get_classifier().in_features

   return model, num_features
