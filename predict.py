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

def run_inference(onnx_model_path, image_folder, image_width, image_height):
   sessions = onnxruntime.InferenceSession(onnx_model_path)
   
   input_name = sessions.get_inputs()[0].name
   
   _transforms = transforms.Compose([
         transforms.Resize((image_width, image_height)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.370, 0.133, 0.092], std=[0.327, 0.090, 0.105])
   ])
   softmax = torch.nn.Softmax(dim=0)
   predictions = []
   for image_path in tqdm(os.listdir(image_folder), desc=f"Processing images in {image_folder}"):
      image = Image.fromarray(cv2.cvtColor(cv2.imread(os.path.join(image_folder, image_path)), cv2.COLOR_BGR2LAB))
      image = _transforms(image)
      
      output = sessions.run(None, {input_name: image.unsqueeze(0).numpy()})
      probs = softmax(torch.tensor(output[0][0]))
      predicted_class = np.argmax(probs)
      predictions.append({
            'image_name': image_path,
            'target': int(predicted_class)
        })
      
   return pd.DataFrame(predictions)
      
predictions = run_inference(
   r"C:\lumen_melanoma_classification\melanoma-classification\weights\12_best_model_domain_discriminative\model_onnx.onnx",
   r"C:\lumen_melanoma_classification\melanoma-classification\isic2020_challenge\valid\malignant",
   224,
   224
)

predictions.to_csv(r"C:\lumen_melanoma_classification\melanoma-classification\weights\12_best_model_domain_discriminative\predictions.csv", index=False)
   