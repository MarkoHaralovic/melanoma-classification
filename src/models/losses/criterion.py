from torch import long, nn
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
      
      gt_counter = torch.ones((self.n_classes,), device=targets.device)
      gt_idx, gt_count = torch.unique(targets, return_counts=True)
      
      # gt_count[gt_idx==self.ignore_index] = gt_count[1].clone()
      # gt_idx[gt_idx==self.ignore_index] = 1 
      gt_counter[gt_idx] = gt_count.float()
      
      fn_counter = torch.ones((self.n_classes), device=targets.device)
      fn = targets.view(-1)[idex]
      fn_idx, fn_count = torch.unique(fn, return_counts=True)
      
      # fn_count[fn_idx==self.ignore_index] = fn_count[1].clone()
      # fn_idx[fn_idx==self.ignore_index] = 1 
      fn_counter[fn_idx] = fn_count.float()
      
      weight = fn_counter / gt_counter
      
      CE = self.criterion(logits, targets)
      
      loss =  weight[targets] * CE # weight the loss based on the recall of each class, bigger weight for lower recall
      
      return loss.mean()
        
class DomainIndependentLoss(nn.Module):
   def __init__(self, num_classes, num_domains, weight=None, conditional_accuracy=False):
      super(DomainIndependentLoss, self).__init__()
      self.num_classes = num_classes
      self.criterion = F.nll_loss
      self.weight = weight
      self.num_domains = num_domains
      self.conditional_accuracy = conditional_accuracy
      
   def forward(self, logits, targets, groups):
      domain_dependant_targets = 2 * groups + targets
      output = []
      for i in range(self.num_domains):
         domain_logits = logits[:, i * self.num_classes:(i + 1) * self.num_classes]
         log_probs = F.log_softmax(domain_logits, dim=1)
         output.append(log_probs)
      logits = torch.cat(output, dim=1)
      loss = self.criterion(logits, domain_dependant_targets, weight=self.weight)
      return loss
   
   
class DomainDiscriminativeLoss(nn.Module):
   def __init__(self, num_classes, num_domains, weight=None, ):
      super(DomainDiscriminativeLoss, self).__init__()
      self.criterion = F.cross_entropy
      self.num_classes = num_classes
      self.num_domains = num_domains
      self.weight = weight
      
   def forward(self, logits, targets):
      return self.criterion(logits, targets, weight=self.weight)
   def get_probs(self, logits):
      return F.softmax(logits, dim=1).detach()
    
class FocalLoss(nn.Module):
   def __init__(self, gamma=0, alpha=None, size_average=True):
      super(FocalLoss, self).__init__()
      self.gamma = gamma
      self.alpha = alpha
      if isinstance(alpha, (float, int)): 
         self.alpha = torch.Tensor([alpha, 1-alpha])
      if isinstance(alpha, list): 
         self.alpha = torch.Tensor(alpha)
      self.size_average = size_average

   def forward(self, input, target):
      target = target.view(-1, 1)

      logpt = F.log_softmax(input, dim=1)
      logpt = logpt.gather(1, target)
      logpt = logpt.view(-1)
      pt = logpt.data.exp()

      if self.alpha is not None:
         if self.alpha.type() != input.data.type():
               self.alpha = self.alpha.to(input.device)
         at = self.alpha.gather(0, target.view(-1))
         logpt = logpt * at

      loss = -1 * (1-pt)**self.gamma * logpt
      if self.size_average: 
         return loss.mean()
      else: 
         return loss.sum()
        
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
      weights = alpha * np.sum(weights) / weights
      
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