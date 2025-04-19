from tqdm import tqdm
import torch
import numpy as np

def compute_cielab_stats(data_loader):
   """Compute mean and std for CIELAB images in a dataset"""
   l_sum, a_sum, b_sum = 0.0, 0.0, 0.0
   l_sq_sum, a_sq_sum, b_sq_sum = 0.0, 0.0, 0.0
   num_pixels = 0
   
   print("Computing CIELAB statistics...")
   for images, _, _ in tqdm(data_loader):
      batch_size = images.size(0)
      num_pixels += batch_size * images.size(2) * images.size(3)
      
      l_sum += torch.sum(images[:, 0, :, :]).item()
      a_sum += torch.sum(images[:, 1, :, :]).item()
      b_sum += torch.sum(images[:, 2, :, :]).item()
      
      l_sq_sum += torch.sum(images[:, 0, :, :] ** 2).item()
      a_sq_sum += torch.sum(images[:, 1, :, :] ** 2).item()
      b_sq_sum += torch.sum(images[:, 2, :, :] ** 2).item()
   
   l_mean = l_sum / num_pixels
   a_mean = a_sum / num_pixels
   b_mean = b_sum / num_pixels
   
   l_std = np.sqrt((l_sq_sum / num_pixels) - (l_mean ** 2))
   a_std = np.sqrt((a_sq_sum / num_pixels) - (a_mean ** 2))
   b_std = np.sqrt((b_sq_sum / num_pixels) - (b_mean ** 2))
   
   return [l_mean, a_mean, b_mean], [l_std, a_std, b_std]

def ita_to_group(ita):
   if ita > 55:
      # Very light
      return 0
   elif ita > 41:
      # Light
      return 1
   elif ita > 28:
      # Intermediate
      return 2
   elif ita > 10:
      # Tan
      return 3
   elif ita > -30:
      # Brown
      return 3
   else:
      # Dark
      return 3 