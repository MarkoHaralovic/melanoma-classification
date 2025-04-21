import str2bool
import torchvision.transforms as transforms
from src.datasets.datasets import LocalISICDataset
import matplotlib.pyplot as plt
import torch
import argparse
import numpy as np

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
     
def main(args):
   if not args.cielab:
        transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
   else:
      transform = transforms.Compose([
         transforms.Resize((args.input_size, args.input_size)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.370, 0.133, 0.092], std=[0.327, 0.090, 0.105])
      ])
   malignant_class_transform = {
            "original": transforms.Compose([]),
            "horizontal_flip": transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)]),
            "vertical_flip": transforms.Compose([transforms.RandomVerticalFlip(p=1.0)]),
            "rotate": transforms.Compose([transforms.RandomRotation(15)]),
            "translate": transforms.Compose([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))])
   }
      
   dataset = LocalISICDataset(args.data_path, 
                              transform=transform,
                              augment_transforms = malignant_class_transform,
                              split='valid',
                              skin_color_csv=args.skin_color_csv,
                              cielab=args.cielab,
                              skin_former=args.skin_former,
                              segment_out_skin=args.segment_out_skin,
                              )
   dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=args.pin_mem,
        drop_last=True
    )
   
   for i in range(max(1, int(args.visualize_num // args.batch_size))):
      images, targets, groups = next(iter(dataloader))
      fig, axes = plt.subplots(1, args.batch_size, figsize=(15, 5))
      for j in range(args.batch_size):
         img = images[j].numpy() if not isinstance(images[j], np.ndarray) else images[j]

         if not args.cielab:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
         else:
            mean = np.array([0.370, 0.133, 0.092])
            std = np.array([0.327, 0.090, 0.105])

         mean = mean.reshape(3, 1, 1)  
         std = std.reshape(3, 1, 1)
         
         img = std * img + mean
         img = np.clip(img, 0, 1) 

         img = np.transpose(img, (1, 2, 0))
         axes[j].imshow(img)
         axes[j].set_title(f'Label: {targets[j].item()}, Group: {groups[j].item()}')
         axes[j].axis('off')
      plt.show()
      
if __name__ == '__main__':
   parser = argparse.ArgumentParser('Melanoma imaes visualization')

   parser.add_argument('--data_path', default='./isic2020_challenge', type=str,
                     help='Path to dataset with train/valid folders')
   parser.add_argument('--input_size', default=224, type=int)
   parser.add_argument('--skin_color_csv', default="./isic2020_challenge/ISIC_2020_full.csv", type=str,
                        help='Path to the CSV file containing skin color labels')
   
   parser.add_argument('--visualize_num', default=5, type=int,
                     help='Number of images to visualize')
   parser.add_argument('--batch_size', default=5, type=int,
                     help='Batch size for visualization')
   
   # Augmentation parameters
   parser.add_argument('--cielab', action='store_true', default=True,
                     help='Load images to CIELab colorspace')
   parser.add_argument('--skin_former', action='store_true', default=False,
                     help='Transform lighter skin types to darker ones')
   parser.add_argument('--segment_out_skin', type=str2bool, default=True,
                     help='Segment out skin from images')
  
   parser.add_argument('--device', default='cpu', type=str,
                     help='Device to use for training')
   parser.add_argument('--pin_mem', type=str2bool, default=True,
                     help='Pin CPU memory for data loading')
   args = parser.parse_args()

   print(args)
   main(args)