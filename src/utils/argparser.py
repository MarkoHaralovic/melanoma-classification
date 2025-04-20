import argparse
import yaml
import os
import logging
from copy import deepcopy

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
     
def parse_args():
    parser = argparse.ArgumentParser('Melanoma Classification Training')
        
    parser.add_argument('--kaggle', action='store_true', default=False,
                    help='Runnin on kaggle')
    parser.add_argument('--kaggle_csv_file', default=r"/kaggle/input/siim-isic-melanoma-classification/train.csv", type=str,
                        help='Path to the CSV file containing image names and labels on Kaggle')
    parser.add_argument('--kaggle_image_dir', default=r"/kaggle/input/siim-isic-melanoma-classification/jpeg/train", type=str,
                        help='Path to the folder with images on Kaggle')
    parser.add_argument('--skin_color_csv', default=None, type=str,
                        help='Path to the CSV file containing skin color labels')

    parser.add_argument('--data_path', default='./isic2020_challenge', type=str,
                        help='Path to dataset with train/valid folders')
    parser.add_argument('--output_dir', default='./melanoma_output', type=str,
                        help='Path to save model outputs')
    parser.add_argument('--log_dir', default='./melanoma_logs', type=str,
                        help='Path for tensorboard logs')

    parser.add_argument('--model', default='convnext_tiny', type=str,
                        choices=['convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large',
                        'efficientnetv2_b0', 'efficientnetv2_b1', 'efficientnetv2_b2', 'efficientnetv2_b3', 
                        'efficientnetv2_m', 'efficientnetv2_l', 'efficientnetv2_s', 'efficientnetv2_xl',
                        'dinov2_vit_base', 'dinov2_vit_small', 'dinov2_vit_large', 'dinov2_vit_giant2',
                        'convnextv2_atto', 'convnextv2_femto', 'convnextv2_pico', 'convnextv2_nano',
                        'convnextv2_tiny', 'convnextv2_small', 'convnextv2_base', 'convnextv2_large',
                        'convnextv2_xlarge'
                        ],
                        help='Model architecture to use')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='Number of output classes')
    parser.add_argument('--num_groups', default = 1, type=int,
                        help = "Number of skin color groups")
    parser.add_argument('--pretrained', default=True, type=str2bool,
                        help='Use pretrained weights')
    parser.add_argument('--freeze_model', default=False, type=str2bool,
                        help='Freeze model weights')
    parser.add_argument('--in_22k', default=False, type=str2bool,
                        help='Use pretrained weights on ImageNet22k')
    parser.add_argument('--input_size', default=224, type=int,
                        help='Input image size')
    parser.add_argument('--drop_path', type=float, default=0.1,
                        help='Drop path rate')

    # Perform evaluation only    
    parser.add_argument('--test', action='store_true', default=False,
                        help='Perform evaluation only')

    # Training parameter
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
    parser.add_argument('--focal_loss', action='store_true', default=False,
                        help='Use Focal loss')
    parser.add_argument('--domain_independent_loss', action='store_true', default=False,
                        help='Use Domain Independent Loss')
    parser.add_argument('--domain_discriminative_loss', action='store_true', default=False,
                        help='Use Domain Discriminative Loss')
    parser.add_argument('--conditional_accuracy', type=str2bool)

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Lower LR bound for cyclic schedulers')
    parser.add_argument('--weight_decay', default=5e-3, type=float,
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
    parser.add_argument('--oversample_malignant', action='store_true', default=False,
                        help='Oversample malignant lesions')
    parser.add_argument('--undersample_benign', action='store_true', default=False,
                        help='Undersample benign lesions')
    parser.add_argument('--undersample_benign_ratio', type=float, default=-1,
                        help='Undersample benign lesions')

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
    parser.add_argument('--cielab', action='store_true', default=False,
                        help='Load images to CIELab colorspace')
    parser.add_argument('--skin_former', action='store_true', default=False,
                        help='Transform lighter skin types to darker ones')
    parser.add_argument('--segment_out_skin', type=str2bool, default=False,
                        help='Use skin segmentation to remove background')
    # Model EMA parameters
    parser.add_argument('--model_ema', type=str2bool, default=False,
                        help='Enable tracking moving average of model weights')
    parser.add_argument('--model_ema_decay', type=float, default=0.9999,
                        help='Factor for model weights moving average')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False,
                        help='Force EMA model weights to be stored on CPU')

    # Misc parameters
    parser.add_argument('--save_ckpt', type=str2bool, default=True,
                        help='Save checkpoints during training')
    parser.add_argument('--save_ckpt_freq', type=int, default=1,
                        help='Frequency to save checkpoints (epochs)')
    parser.add_argument('--seed', default=0, type=int,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device to use for training')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers for data loading')
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory for data loading')
    parser.add_argument('--checkpoint', default='', type=str,
                        help='Path to checkpoint training/evaluation.') 
            
   # distributed training parameters 
    parser.add_argument('--use_amp', type=str2bool, default=False,
                    help='Use PyTorch automatic mixed precision')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--distributed', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    # add config file
    parser.add_argument('--config', default=None, type=str,
                        help='Path to the config file')
    
    args = parser.parse_args()
    if args.weight_decay_end is None:
            args.weight_decay_end = args.weight_decay
            
    if args.config is not None and os.path.exists(args.config):
        config = load_yaml_config(args.config)
        args = update_args_with_config(args, config)
    return args

def load_yaml_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def update_args_with_config(args, config_dict):
    parser, explicit_args = args._get_kwargs(), {}
    
    for k, v in parser:
        if hasattr(args, f"__{k}"):
            explicit_args[k] = True
    
    args_copy = deepcopy(args)
    for k, v in config_dict.items():
        if k not in explicit_args and hasattr(args_copy, k):
            setattr(args_copy, k, v)
    
    return args_copy