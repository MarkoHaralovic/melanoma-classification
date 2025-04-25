import torch
import logging
import os
from ..utils.utils import convert_to_torchscript, convert_to_onnx
from ..models.melanoma_classifier import MelanomaClassifier

def convert_checkpoint_to_torchscript_and_onnx(checkpoint_path,
                                               model_class, 
                                               output_dir,
                                               input_size=224):
   """
   Convert a saved checkpoint (.pth file) to TorchScript and ONNX
   """
   if not os.path.exists(output_dir):
      os.makedirs(output_dir)
   
   logging.info(f"Loading checkpoint from {checkpoint_path}")
   checkpoint = torch.load(checkpoint_path, map_location='cpu')
   
   if isinstance(checkpoint, dict) and 'args' in checkpoint:
      chkpt_args = checkpoint["args"]
      
      num_classes = chkpt_args.num_classes
      model_name = chkpt_args.model
      pretrained = False
      in_22k = chkpt_args.in_22k
      if checkpoint['args'].num_groups > 0:
         num_classes = chkpt_args.num_classes * chkpt_args.num_groups
      
      model = model_class(
         model_name=model_name,
         num_classes=num_classes,
         pretrained=pretrained,
         in_22k=in_22k
      )
   else:
      model = model_class(num_classes=2) 

   print(checkpoint.keys())
   if 'model_state_dict' in checkpoint:
      state_dict = checkpoint['model_state_dict']
   else:
      state_dict = checkpoint['model']
   
   model.load_state_dict(state_dict)
   print(model)
   model.eval()
   
   input_tensor = torch.randn(1, 3, input_size, input_size)

   logging.info(f"Converting to TorchScript...")
   convert_to_torchscript(model, input_tensor, os.path.join(output_dir, "model_torchscript.pt"), True)
   logging.info(f"Converting to ONNX...")
   convert_to_onnx(model, input_tensor, os.path.join(output_dir, "model_onnx.onnx"))
   
   logging.info(f"Models exported to {output_dir}")
   
checkpoint_path = r"C:\lumen_melanoma_classification\melanoma-classification\weights\best_model_domain_discriminative.pth"
convert_checkpoint_to_torchscript_and_onnx(
   checkpoint_path=checkpoint_path,
   model_class=MelanomaClassifier,
   input_size=224,
   output_dir=r"C:\lumen_melanoma_classification\melanoma-classification\weights\12_best_model_domain_discriminative"
)