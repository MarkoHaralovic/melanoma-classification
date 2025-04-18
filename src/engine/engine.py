import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from ..models.losses.criterion import DomainIndependentLoss, DomainDiscriminativeLoss
from ..utils import utils

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()

    for data_iter_step, (samples, targets, groups) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        groups = groups.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.amp.autocast("cuda"):
                output = model(samples)
                if isinstance(criterion, DomainIndependentLoss):
                    loss = criterion(output, targets, groups)
                else:
                    loss = criterion(output, targets)
        else: # full precision
            output = model(samples)
            if isinstance(criterion, DomainIndependentLoss):
                loss = criterion(output, targets, groups)
            else:
                loss = criterion(output, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if mixup_fn is None:
            if isinstance(criterion, DomainIndependentLoss):
                preds = utils.compute_preds_sum_out(output, criterion.num_classes, criterion.num_domains)
                class_acc = (preds == targets).float().sum() / targets.shape[0]
            elif isinstance(criterion, DomainDiscriminativeLoss):
                probs = criterion.get_probs(output)
                preds = utils.compute_accuracy_sum_prob_wo_prior_shift(probs, criterion.num_classes, criterion.num_domains)
                class_acc = (preds == targets).float().sum() / targets.shape[0]
            else:
                class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
            
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            if class_acc:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc': class_acc}, commit=False)
            if use_amp:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False, criterion=None):
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    y_true = []
    y_pred = []
    groups = []
    
    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]
        group = batch[2]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        group = group.to(device, non_blocking=True)
        
        # compute output
        if use_amp:
            with torch.amp.autocast("cuda"):
                output = model(images)
                if isinstance(criterion, DomainIndependentLoss):
                    loss = criterion(output, target, group)
                else:
                    loss = criterion(output, target)
        else:
            output = model(images)

            if isinstance(criterion, DomainIndependentLoss):
                loss = criterion(output, target, group)
            else:
                loss = criterion(output, target)

        if isinstance(criterion, DomainIndependentLoss) and not criterion.conditional_accuracy:
            preds = utils.compute_preds_sum_out(output,criterion.num_classes, criterion.num_domains)
            acc1 = (preds == target).float().sum() / target.shape[0]
        elif isinstance(criterion, DomainIndependentLoss) and criterion.conditional_accuracy:
            preds = utils.compute_preds_conditional(output,criterion.num_classes, criterion.num_domains, group)
            acc1 = (preds == target).float().sum() / target.shape[0]
        elif isinstance(criterion, DomainDiscriminativeLoss):
            probs = criterion.get_probs(output)
            preds = utils.compute_accuracy_sum_prob_wo_prior_shift(probs, criterion.num_classes, criterion.num_domains)
            acc1 = (preds == target).float().sum() / target.shape[0]
        else:
            preds = utils._eval(output, target)[0]
            acc1 = accuracy(output, target, topk=(1,5))[0]

        y_true.extend(target.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        groups.extend(group.cpu().tolist())
                
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Global acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))

    malignant_recall, malignant_precision, malignant_f1, malignant_dpd = utils.get_metrics(y_true, y_pred, groups)
    
    metric_logger.meters['malignant_recall'].update(malignant_recall)
    metric_logger.meters['malignant_precision'].update(malignant_precision)
    metric_logger.meters['malignant_f1'].update(malignant_f1)
    metric_logger.meters['malignant_dpd'].update(malignant_dpd)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
