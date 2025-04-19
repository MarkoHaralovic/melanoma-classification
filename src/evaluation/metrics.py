import logging
import torch
import torch.nn.functional as F
import numpy as np
from fairlearn.metrics import demographic_parity_difference

def _eval(output, topk=(1,)):
   """Computes the predictions over the k top predictions for the specified values of k"""
   maxk = min(max(topk), output.size()[1])
   _, pred = output.topk(maxk, 1, True, True)
   pred = pred.t()
   return pred
 
def compute_preds_sum_out(outputs, num_classes, num_domains):
   _logits = []
   for i in range(num_domains):
      _logits.append(outputs[:, i * num_classes:(i + 1) * num_classes])
   _logits = torch.stack(_logits, dim=0).sum(dim=0)
   predictions = torch.argmax(_logits, axis=1)
   
   return predictions

def compute_preds_conditional(outputs, num_classes, num_domains, groups):
   predictions = torch.zeros(outputs.shape[0], dtype=torch.int64).to(outputs.device)
   for i in range(num_domains):
      _ids = (groups == i)
      if _ids.sum() > 0:
         _logits = outputs[_ids, i * num_classes:(i + 1) * num_classes]
         _pred = torch.argmax(_logits, axis=1)
         predictions[_ids] = _pred
   return predictions

def compute_preds_sum_prob_w_prior_shift(outputs, num_classes, num_domains):
   # Training distributions per domain
   prior_shift_weight = np.array([
      1088/1072, 1088/16, 17746/17515, 17746/231, 6454/6273, 6454/181, 850/834, 850/16 
   ]) / 100
   probs = F.softmax(outputs, dim=1).cpu().numpy() * prior_shift_weight
   domain_probs = []
   for i in range(num_domains):
      domain_probs.append(probs[:, i * num_classes:(i + 1) * num_classes])
   summed_probs = torch.stack(domain_probs, dim=0).sum(dim=0)
   predictions = torch.argmax(summed_probs, axis=1)
   return predictions

def get_metrics(y_true, y_pred, groups):
   """
      y_true: list of true labels
      y_pred: list of predicted labels
      groups: list of skin type groups
   """
   y_pred = torch.tensor(y_pred).to(torch.int64)
   y_true = torch.tensor(y_true).to(torch.int64)
   groups = torch.tensor(groups).to(torch.int64)

   correct = y_pred.eq(y_true)
   
   global_acc = correct.float().sum() * 100. / y_true.size(0)
   logging.info(f"Global accuracy: {global_acc.item()}")
   
   tp = ((y_pred == 1) & (y_true == 1)).sum().item()
   fp = ((y_pred == 1) & (y_true == 0)).sum().item()
   tn = ((y_pred == 0) & (y_true == 0)).sum().item()
   fn = ((y_pred == 0) & (y_true == 1)).sum().item()
   
   confusion_matrix = {
      'TP': tp,
      'FP': fp,
      'TN': tn,
      'FN': fn
   }
   logging.info("Confusion Matrix: ", confusion_matrix)
   
   malignant = y_true == 1
   
   malignant_recall = tp / (tp + fn + 1e-10) 
   malignant_precision = tp / (tp + fp + 1e-10) 
   malignant_f1 = 2 * malignant_precision * malignant_recall / (malignant_precision + malignant_recall + 1e-10) 
   
   logging.info(f"Malignant recall: {malignant_recall:.4f}")
   logging.info(f"Malignant precision: {malignant_precision:.4f}")
   logging.info(f"Malignant F1: {malignant_f1:.4f}")

   benign_recall = tn / (tn + fp + 1e-10)
   benign_precision = tn / (tn + fn + 1e-10) 
   benign_f1 = 2 * benign_precision * benign_recall / (benign_precision + benign_recall + 1e-10)
   
   logging.info(f"benign precision: {benign_precision}")
   logging.info(f"benign f1: {benign_f1}")
   
   try:
      overall_dpd = demographic_parity_difference(
               y_true = y_true.cpu().numpy(), 
               y_pred = y_pred.cpu().numpy(), 
               sensitive_features = groups.cpu().numpy()
         )
      logging.info(f"Demographic parity difference on the whole dataset: {overall_dpd.item()}")
      
      malignant_dpd = demographic_parity_difference(
               y_true = y_true[malignant].cpu().numpy(), 
               y_pred = y_pred[malignant].cpu().numpy(), 
               sensitive_features = groups[malignant].cpu().numpy()
         )
      logging.info(f"Demographic parity difference on malignant subset: {malignant_dpd.item()}")
   except Exception as e:
      logging.error("Error calculating demographic parity difference: ", e)
   
   for _group in torch.unique(groups):
      group_y_pred = y_pred[groups == _group]
      group_y_true = y_true[groups == _group]
      
      group_tp = ((group_y_pred == 1) & (group_y_true == 1)).sum().item()
      group_fp = ((group_y_pred == 1) & (group_y_true == 0)).sum().item()
      group_tn = ((group_y_pred == 0) & (group_y_true == 0)).sum().item()
      group_fn = ((group_y_pred == 0) & (group_y_true == 1)).sum().item()
      
      group_malignant_recall = group_tp / (group_tp + group_fn + 1e-10) 
      group_malignant_precision = group_tp / (group_tp + group_fp + 1e-10)
      group_malignant_f1 = 2 * group_malignant_precision * group_malignant_recall / (group_malignant_precision + group_malignant_recall + 1e-10)
      logging.info(f"\nEVALUATION FOR GROUP {_group.item()}:")
      logging.info(f"   Group size: {group_y_true.shape[0]} samples")
      logging.info(f"   Group TP: {group_tp}, FP: {group_fp}, TN: {group_tn}, FN: {group_fn}")
      logging.info(f"   Group malignant recall: {group_malignant_recall:.4f}")
      logging.info(f"   Group malignant precision: {group_malignant_precision:.4f}")
      logging.info(f"   Group malignant F1: {group_malignant_f1:.4f}")
      logging.info("-----------------------------------------------------------------------")
   
   return malignant_recall, malignant_precision, malignant_f1, malignant_dpd
   
   