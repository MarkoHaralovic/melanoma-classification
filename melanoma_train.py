import argparse
from datetime import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
import os
import sys
from pathlib import Path

from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from models.melanoma_classifier import MelanomaClassifier
from models.criterion import OhemCrossEntropy,RecallCrossEntropy, labels_to_class_weights
from engine import train_one_epoch, evaluate
from utils import NativeScalerWithGradNormCount as NativeScaler
from optim_factory import create_optimizer, LayerDecayValueAssigner
import utils
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def train(args):
    train_dir = os.path.join(args.data_path, 'train')
    valid_dir = os.path.join(args.data_path, 'valid')
    logging.info(f"Train directory: {train_dir}")
    logging.info(f"Valid directory: {valid_dir}")
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    
    if not os.path.exists(valid_dir):
        raise FileNotFoundError(f"Valid directory not found: {valid_dir}")
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    
    train_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(valid_dir, transform=val_transform)
    
    logging.info(f"Train dataset classes: {train_dataset.classes}")
    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(val_dataset)}")
    logging.info(f"Training with  {args.batch_size} images per batch")
    logging.info(f"Training with  {args.num_classes} classes")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    log_writer = None
    if args.output_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = args.model
        run_name = f"{timestamp}_{model_name}_bs{args.batch_size}"
        
        args.output_dir = os.path.join(args.output_dir, run_name)
        os.makedirs(args.output_dir, exist_ok=True)
        
        log_file = os.path.join(args.output_dir, "training.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        logging.info(f"Saving outputs to: {args.output_dir}")
        logging.info(f"Logging to: {log_file}")
        
        with open(os.path.join(args.output_dir, "config.json"), 'w') as f:
            json.dump(vars(args), f, indent=4)
        args.log_dir = os.path.join(args.output_dir, "logs")
        
        if args.log_dir:
            os.makedirs(args.log_dir, exist_ok=True)
            log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    
    model = MelanomaClassifier(
        model_name=args.model,
        num_classes=args.num_classes,
        pretrained=args.pretrained
    )
    model = model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)
    
    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else None,
            resume=''
        )
        logging.info(f"Using EMA with decay = {args.model_ema_decay}")
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Number of parameters: {n_parameters}")
    
    mixup_fn = None
    if args.mixup > 0 or args.cutmix > 0 or args.cutmix_minmax is not None:
        logging.info("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, 
            cutmix_alpha=args.cutmix, 
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, 
            switch_prob=args.mixup_switch_prob, 
            mode=args.mixup_mode,
            label_smoothing=args.smoothing, 
            num_classes=args.num_classes
        )
    if args.layer_decay < 1.0 or args.layer_decay > 1.0:
        num_layers = 12 # convnext layers divided into 12 parts, each with a different decayed lr value.
        assert args.model in ['convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge'], \
             "Layer Decay impl only supports convnext_small/base/large/xlarge"
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))
        
    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=None,
        get_num_layer=assigner.get_layer_id if assigner is not None else None, 
        get_layer_scale=assigner.get_scale if assigner is not None else None)
    
    loss_scaler = NativeScaler() if args.use_amp else None
    
    num_training_steps_per_epoch = len(train_dataset) // args.batch_size
    
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    logging.info(f"Max WD = {max(wd_schedule_values):.7f}, Min WD = {min(wd_schedule_values):.7f}")
    
    if args.ifw:
        class_weights = labels_to_class_weights(train_dataset.samples, args.num_classes, alpha = 0.5)
    else:
        class_weights = torch.ones(args.num_classes) 
        
    if args.ohem:
        criterion = OhemCrossEntropy(
            ignore_label=-1, 
            thres=0.7, 
            weight = class_weights)  
    elif args.recall_ce:
        criterion = RecallCrossEntropy(
            n_classes=args.num_classes, 
            weight = class_weights)
    elif mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight = class_weights)
    
    logging.info(f"Criterion: {criterion}")
    
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info(f"Loading checkpoint from: {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            args.start_epoch = checkpoint.get('epoch', 0) + 1
            logging.info(f"Loaded checkpoint '{args.resume}' (epoch {args.start_epoch-1})")
        else:
            logging.info(f"No checkpoint found at: {args.resume}")
    
    max_accuracy = 0.0
    best_epoch = 0
    
    logging.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        logging.info(f"\nEpoch {epoch+1}/{args.epochs}")
        logging.info('-' * 30)
        
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        
        model.train()
        train_stats = train_one_epoch(
            model, criterion, train_loader, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            use_amp=args.use_amp
        )
        
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                save_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'args': args,
                }, save_path)
                logging.info(f"Saved checkpoint to: {save_path}")
        
        model.eval()
        test_stats = evaluate(val_loader, model, device, use_amp=args.use_amp)
        logging.info(f"Validation accuracy: {test_stats['acc1']:.2f}%")
        
        if test_stats['acc1'] > max_accuracy:
            max_accuracy = test_stats['acc1']
            best_epoch = epoch
            if args.output_dir and args.save_ckpt:
                save_path = os.path.join(args.output_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': max_accuracy,
                    'args': args,
                }, save_path)
                logging.info(f"Saved new best model with validation accuracy: {max_accuracy:.2f}%")
        
        logging.info(f"Current best accuracy: {max_accuracy:.2f}% (epoch {best_epoch+1})")
        
        if log_writer is not None:
            log_writer.update(test_acc1=test_stats['acc1'], head="perf", step=epoch)
            log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)
        
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters
        }
        
        if args.output_dir:
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f'\nTraining completed in {total_time_str}')
    logging.info(f'Best validation accuracy: {max_accuracy:.2f}% (epoch {best_epoch+1})')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Melanoma Classification Training')
    
    parser.add_argument('--data_path', default='./isic2020_challenge', type=str,
                        help='Path to dataset with train/valid folders')
    parser.add_argument('--output_dir', default='./melanoma_output', type=str,
                        help='Path to save model outputs')
    parser.add_argument('--log_dir', default='./melanoma_logs', type=str,
                        help='Path for tensorboard logs')
    
    parser.add_argument('--model', default='convnext_tiny', type=str,
                        choices=['convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large'],
                        help='Model architecture to use')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='Number of output classes')
    parser.add_argument('--pretrained', default=True, type=str2bool,
                        help='Use pretrained weights')
    parser.add_argument('--input_size', default=224, type=int,
                        help='Input image size')
    parser.add_argument('--drop_path', type=float, default=0.1,
                        help='Drop path rate')
    
    # Training parameters
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size for training and validation')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Number of epochs to train')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='Start epoch for resuming training')
    parser.add_argument('--update_freq', default=1, type=int,
                        help='Gradient accumulation steps')
    
    # Criterion parameters
    parser.add_argument('--ohem', action='store_true', default=False,
                    help='Use OHEM loss')
    parser.add_argument('--ifw', action='store_true', default=False,
                        help='Use inverse frequency weighting.')
    parser.add_argument('--recall_ce', action='store_true', default=False,
                        help='Use Recall Cross Entropy loss')
    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Lower LR bound for cyclic schedulers')
    parser.add_argument('--weight_decay', default=0.05, type=float,
                        help='Weight decay')
    parser.add_argument('--weight_decay_end', default=None, type=float,
                        help='Final weight decay value')
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='Clip gradient norm value (None = no clipping)')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Epochs to warmup LR')
    parser.add_argument('--warmup_steps', type=int, default=-1,
                        help='Steps to warmup LR (overrides warmup_epochs if > 0)')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    
    # Augmentation parameters
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing value')
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of applying mixup/cutmix')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix ["batch", "pair", "elem"]')
    
    # Model EMA parameters
    parser.add_argument('--model_ema', type=str2bool, default=False,
                        help='Enable tracking moving average of model weights')
    parser.add_argument('--model_ema_decay', type=float, default=0.9999,
                        help='Factor for model weights moving average')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False,
                        help='Force EMA model weights to be stored on CPU')
    
    # Misc parameters
    parser.add_argument('--use_amp', type=str2bool, default=False,
                        help='Use PyTorch automatic mixed precision')
    parser.add_argument('--save_ckpt', type=str2bool, default=True,
                        help='Save checkpoints during training')
    parser.add_argument('--save_ckpt_freq', type=int, default=5,
                        help='Frequency to save checkpoints (epochs)')
    parser.add_argument('--seed', default=0, type=int,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='Device to use for training')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers for data loading')
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory for data loading')
    parser.add_argument('--resume', default='', type=str,
                        help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    
    train(args)