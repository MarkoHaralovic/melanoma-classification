import os
import logging
import argparse

import torch
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

from src.datasets.datasets import LocalISICDataset
from src.models.melanoma_classifier import MelanomaClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger()

@torch.no_grad()
def eval(args):
    test_dir = os.path.join(args.data_path, args.split)
    logging.info(f"Test directory: {test_dir}")

    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    checkpoint = torch.load(
        os.path.join(args.model_dir, args.checkpoint), weights_only=False
    )
    logging.info(
        f"Loaded checkpoint '{args.checkpoint}' from {args.model_dir} (epoch {checkpoint['epoch']})"
    )
    chkpt_args = checkpoint["args"]

    # Data
    if not getattr(chkpt_args, "cielab", False):
        transform = transforms.Compose(
            [
                transforms.Resize((chkpt_args.input_size, chkpt_args.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((chkpt_args.input_size, chkpt_args.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.370, 0.133, 0.092], std=[0.327, 0.090, 0.105]
                ),
            ]
        )

    test_dataset = LocalISICDataset(
        args.data_path,
        transform=transform,
        split=args.split,
        augment_transforms=None,
        skin_former=False,
        skin_color_csv=args.skin_color_csv,
        cielab=getattr(chkpt_args, "cielab", False),
        segment_out_skin=getattr(chkpt_args, "segment_out_skin", False),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        shuffle=False,
        sampler=None,
        drop_last=False,
    )

    logging.info(f"Test dataset size: {len(test_dataset)}")
    logging.info(f"Test dataset classes: {test_dataset.classes}")
    logging.info(f"Test dataset class_distribution: {test_dataset.class_distribution}")
    logging.info(f"Test dataset groups: {test_dataset.groups}")
    logging.info(f"Test dataset group distribution: {test_dataset.group}")

    # Model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = MelanomaClassifier(
        model_name=chkpt_args.model,
        num_classes=chkpt_args.num_classes * getattr(chkpt_args, "num_groups", 1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Collect outputs
    y_true = []
    logits = []
    groups = []

    for batch in tqdm(test_loader, "Evaluating"):
        images = batch[0]
        target = batch[1]
        group = batch[2]

        images = images.to(device, non_blocking=True)
        output = model(images)
        
        y_true.append(target)
        logits.append(output.cpu())
        groups.append(group)

    y_true = torch.cat(y_true).numpy()
    logits = torch.cat(logits).numpy()
    groups = torch.cat(groups).numpy()

    # Save to model_dir/eval
    save_dir = os.path.join(args.model_dir, "eval", args.checkpoint.split(".")[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logging.info(f"Saving results to {save_dir}")
    np.save(os.path.join(save_dir, "y_true.npy"), y_true)
    np.save(os.path.join(save_dir, "logits.npy"), logits)
    np.save(os.path.join(save_dir, "groups.npy"), groups)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Melanoma Classification Evaluation")

    parser.add_argument("--model_dir", help="Model save directory")
    parser.add_argument(
        "--checkpoint", default="best_model.pth", help="Checkpoint filename"
    )
    parser.add_argument("--split", help="Data subset to evaluate on")
    parser.add_argument(
        "--batch_size", default=16, type=int, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_workers", default=4, type=int, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        default=True,
        help="Pin CPU memory for data loading",
    )
    parser.add_argument(
        "--skin_color_csv",
        default=None,
        type=str,
        help="Path to the CSV file containing skin color labels",
    )
    parser.add_argument(
        "--data_path", type=str, help="Path to dataset with train/valid/test folders"
    )
    parser.add_argument(
        "--device", default="cuda:0", type=str, help="Device to use for evaluation"
    )

    args = parser.parse_args()
    eval(args)
