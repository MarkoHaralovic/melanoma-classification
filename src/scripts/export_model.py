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
   

import argparse

def parse_args():
   parser = argparse.ArgumentParser(description="Convert PyTorch checkpoint to TorchScript and ONNX formats")
   
   parser.add_argument(
      "--checkpoint_path", 
      type=str, 
      default=r"../../weights/best_model_domain_discriminative/best_model.pth",
      help="Path to the checkpoint (.pth file) to be converted"
   )
   
   parser.add_argument(
      "--output_dir", 
      type=str, 
      default=r"../../output/best_model_domain_discriminative",
      help="Directory where the exported models will be saved"
   )
   
   parser.add_argument(
      "--input_size", 
      type=int, 
      default=224,
      help="Input image size (height and width) for the model"
   )
   
   parser.add_argument(
      "--export_torchscript", 
      action="store_true", 
      default=True,
      help="Export model to TorchScript format"
   )
   
   parser.add_argument(
      "--export_onnx", 
      action="store_true", 
      default=True,
      help="Export model to ONNX format"
   )
   
   return parser.parse_args()


if __name__ == "__main__":
   args = parse_args()
   
   log_level = logging.DEBUG if args.verbose else logging.INFO
   logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
   
   convert_checkpoint_to_torchscript_and_onnx(
      checkpoint_path=args.checkpoint_path,
      model_class=MelanomaClassifier,
      input_size=args.input_size,
      output_dir=args.output_dir
   )