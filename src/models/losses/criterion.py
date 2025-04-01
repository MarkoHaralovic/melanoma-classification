from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

class OhemCrossEntropy(nn.Module):
   """Ohem cross entropy

   Args:
      ignore_label (int): ignore label
      thres (float): maximum probability of prediction to be ignored
      min_kept (int): maximum number of pixels to be consider to compute loss
      weight (torch.Tensor): weight for cross entropy loss
   """

   def __init__(self, ignore_label=-1, thres = 0.5,  weight=None):
      super(OhemCrossEntropy, self).__init__()
      self.top_k = thres
      self.ignore_label = ignore_label
      self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction="none")

   def forward(self, logits, targets):
      if self.ignore_label != -1:
            valid_mask = targets != self.ignore_label
            if not valid_mask.all():
                logits = logits[valid_mask]
                targets = targets[valid_mask]
      if logits.shape[0] == 0:
         return logits.sum()
   
      losses = self.criterion(logits, targets)
                
      if isinstance(self.top_k, float):
         k = int(self.top_k * losses.shape[0])  
      else:
         k = self.top_k
      _, indices = torch.topk(losses, k=k, largest = True)
      ohem_loss = losses[indices]
      
      return ohem_loss.mean()
   
class RecallCrossEntropy(nn.Module):
   def __init__(self, n_classes, weight = None, ignore_index = -1):
      super(RecallCrossEntropy, self).__init__()
      self.n_classes = n_classes
      self.criterion = nn.CrossEntropyLoss(reduction='none', weight=weight, ignore_index=ignore_index)
      self.ignore_index = ignore_index
      
   def forward(self, logits, targets):
      pred = logits.argmax(dim=1)
      idex =  (pred != targets).view(-1) # get the index of the misclassified samples
      
      gt_counter = torch.ones((self.n_classes,))
      gt_idx, gt_count = torch.unique(targets, return_counts=True)
      
      gt_count[gt_idx==self.ignore_index] = gt_count[1]
      gt_idx[gt_idx==self.ignore_index] = 1 
      gt_counter[gt_idx] = gt_count.float()
      
      fn_counter = torch.ones((self.n_classes))
      fn = targets.view(-1)[idex]
      fn_idx, fn_count = torch.unique(fn, return_counts=True)
      
      fn_count[fn_idx==self.ignore_index] = fn_count[1]
      fn_idx[fn_idx==self.ignore_index] = 1 
      fn_counter[fn_idx] = fn_count.float()
      
      weight = fn_counter / gt_counter
      
      CE = self.criterion(logits, targets)
      
      loss =  weight[targets] * CE # weight the loss based on the recall of each class, bigger weight for lower recall
      
      return loss.mean()
        
def labels_to_class_weights(samples, ifw_by_skin_type = False, num_classes=2, alpha=0.5, beta=0.5):
   """
   Calculate class weights based on class frequencies in the dataset, taking into account both
   targets and skin color types. Higher weights are assigned to rare combinations.
   
   Args:
      samples: Dataset samples with (path, target, skin_color)
      num_classes: Number of target classes
      alpha: Weight for inverse frequency weighting
      beta: Additional weight factor for rare skin colors with malignant samples
    """
   if not ifw_by_skin_type:
      labels = [sample[1] for sample in samples]
    
      class_counts = np.bincount(labels, minlength=num_classes)
      logging.info(f"Original (not oversampled) benign/malignant class distribution: {class_counts}")
      
      weights = np.copy(class_counts)
      weights[weights == 0] = 1
      weights = 1.0 / weights * alpha
      
   else:
      unique_skin_colors = set()
      for sample in samples:
         unique_skin_colors.add(sample[2])
      
      skin_color_mapping = {color: idx for idx, color in enumerate(unique_skin_colors)}
      num_skin_colors = len(unique_skin_colors)
      logging.info(f"Found skin color types: {skin_color_mapping}")
      
      skin_colors = [skin_color_mapping[sample[2]] for sample in samples]
      skin_color_counts = np.bincount(skin_colors, minlength=num_skin_colors) 
      logging.info(f"Original (not oversampled) skin color class distribution: {skin_color_counts}")
      
      combined_features = [(sample[1], skin_color_mapping[sample[2]]) for sample in samples]
      unique_combinations = set(combined_features)
      
      combination_counts = {}
      for combo in unique_combinations:
         combination_counts[combo] = combined_features.count(combo)
      
      logging.info(f"Target-skin color combination counts: {combination_counts}")
      
      class_weights = np.zeros(num_classes * num_skin_colors)
      
      for combo in unique_combinations:
         target, skin_color = combo
         combo_count = combination_counts[combo]
         combo_idx = target * num_skin_colors + skin_color
         
         weight_value = 1.0 / max(combo_count, 1)
         
         if target == 1:
               weight_value *= (1.0 + beta)
               
         class_weights[combo_idx] = weight_value
      
      class_weights[class_weights == 0] = 0.01
      
      weights = class_weights / class_weights.sum()

   logging.info(f"Final class weights: {weights}")
   return torch.Tensor(weights)