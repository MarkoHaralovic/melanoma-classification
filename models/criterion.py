from torch import nn
from torch.nn import functional as F
import torch
import numpy as np

class OhemCrossEntropy(nn.Module):
   """Ohem cross entropy

   Args:
      ignore_label (int): ignore label
      thres (float): maximum probability of prediction to be ignored
      min_kept (int): maximum number of pixels to be consider to compute loss
      weight (torch.Tensor): weight for cross entropy loss
   """

   def __init__(self, ignore_label=-1, thres=0.7, min_kept=100000, weight=None):
      super(OhemCrossEntropy, self).__init__()
      self.thresh = thres
      self.min_kept = max(1, min_kept)
      self.ignore_label = ignore_label
      self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction="none")

   def forward(self, score, target, **kwargs):
      ph, pw = score.size(2), score.size(3)
      h, w = target.size(1), target.size(2)
      if ph != h or pw != w:
         score = F.upsample(input=score, size=(h, w), mode="bilinear", align_corners=False)

      pred = F.softmax(score, dim=1)
      pixel_losses = self.criterion(score, target).contiguous().view(-1)
      mask = target.contiguous().view(-1) != self.ignore_label

      tmp_target = target.clone()
      tmp_target[tmp_target == self.ignore_label] = 0
      pred = pred.gather(1, tmp_target.unsqueeze(1))
      pred, ind = (
         pred.contiguous()
         .view(
               -1,
         )[mask]
         .contiguous()
         .sort()
      )
      min_value = pred[min(self.min_kept, pred.numel() - 1)]
      threshold = max(min_value, self.thresh)

      pixel_losses = pixel_losses[mask][ind]
      pixel_losses = pixel_losses[pred < threshold]
      return pixel_losses.mean()
   
def labels_to_class_weights(labels, num_classes=80): 
   # Get class weights (inverse frequency) from training labels 
   labels = np.concatenate(labels, 0)  
   classes = labels[:, 0].astype(np.int)  # labels = [class xywh] 
   weights = np.bincount(classes, minlength=num_classes)  # occurences per class 
   weights[weights == 0] = 1  
   weights = (1 / weights) / weights.sum()  # inverse and normalize
   return torch.Tensor(weights) 