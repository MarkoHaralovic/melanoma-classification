import numpy as np
import onnxruntime 
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
import cv2
import pandas as pd
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import argparse

def compute_preds_sum_prob_w_prior_shift(outputs, num_classes, num_domains):
   # Training distributions per domain
   prior_shift_weight = torch.tensor([
      1088/1072, 1088/16, 17746/17515, 17746/231, 6454/6273, 6454/181, 850/834, 850/16 
   ], device=outputs.device) / 100
   probs = F.softmax(outputs, dim=0) * prior_shift_weight
   domain_probs = []
   for i in range(num_domains):
      domain_probs.append(probs[:, i * num_classes:(i + 1) * num_classes])
   summed_probs = torch.stack(domain_probs, dim=0).sum(dim=0)
   predictions = torch.argmax(summed_probs, axis=1)
   return predictions

def run_inference(onnx_model_path, image_folder, image_width, image_height):
   sessions = onnxruntime.InferenceSession(onnx_model_path)
   print(f"Model loaded from {onnx_model_path}")
   
   input_name = sessions.get_inputs()[0].name
   
   _transforms = transforms.Compose([
         transforms.Resize((image_width, image_height)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.370, 0.133, 0.092], std=[0.327, 0.090, 0.105])
   ])
   predictions = []
   for image_path in tqdm(os.listdir(image_folder), desc=f"Processing images in {image_folder}"):
      image = Image.fromarray(cv2.cvtColor(cv2.imread(os.path.join(image_folder, image_path)), cv2.COLOR_BGR2LAB))
      image = _transforms(image)
      output = sessions.run(None, {input_name: image.unsqueeze(0).numpy()})
      preds = compute_preds_sum_prob_w_prior_shift(torch.tensor(output), 2, 4)

      predictions.append({
            'image_name': image_path,
            'target': int(preds)
        })
      
   return pd.DataFrame(predictions)
     
def main():
   parser = argparse.ArgumentParser(description="Run inference on images using an ONNX model.")
   parser.add_argument("--onnx_model_path", type=str, required=False, default = "./weights/best_model.onnx",
                       help="Path to the ONNX model.")
   parser.add_argument("--image_folder", type=str, required=True, 
                       defauklt = "./isic2020_challenge/valid/malignant", 
                       help="Path to the folder containing images.")
   parser.add_argument("--image_width", type=int, default=224, help="Width of the input images.")
   parser.add_argument("--image_height", type=int, default=224, help="Height of the input images.")
   parser.add_argument("--output_csv", type=str, default='/predictions/pred.csv', 
                       help="Path where to save predictions csv..")
   
   args = parser.parse_args()
   
   predictions = run_inference(
      args.onnx_model_path, 
      args.image_folder, 
      args.image_width, 
      args.image_height
   )
   predictions.to_csv(args.output_csv, index=False)