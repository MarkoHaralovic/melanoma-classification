# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2

def build_dataset(is_train, args, transform=None):
    if not args.convert_to_ffcv and transform is None:
        transform = build_transform(is_train, args)
        print("Transform = ")
        if isinstance(transform, tuple):
            for trans in transform:
                print(" - - - - - - - - - - ")
                for t in trans.transforms:
                    print(t)
        else:
            for t in transform.transforms:
                print(t)
        print("---------------------------")

    else:
        import warnings
        warnings.warn("As the transformations are built using factory method timm.data.create_transform,\
                      such a factory method should be implemented for ffcv module as well.\
                      Currently transformations cannot be dynamically allocated for ffcv module \
                      and any ffcv dataset that is built will contain no augmented images, if not specified here otherwise.",
                      UserWarning)
    if args.data_set == 'CIFAR':
        if not args.convert_to_ffcv :
            dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        else:
            dataset = datasets.CIFAR100(args.data_path, train=is_train, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        if not args.convert_to_ffcv :
            dataset = datasets.ImageFolder(root, transform=transform)
        else:
            dataset = datasets.ImageFolder(root)
        nb_classes = 1000
    elif args.data_set == 'IMAGENET1K':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        if not args.convert_to_ffcv :
            dataset = datasets.ImageFolder(root, transform=transform)
        else:
            dataset = datasets.ImageFolder(root)
        nb_classes = 1000
        assert args.nb_classes == nb_classes
    elif args.data_set == 'IMAGENET100':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        if not args.convert_to_ffcv :
            dataset = datasets.ImageFolder(root, transform=transform)
        else:
            dataset = datasets.ImageFolder(root)
        nb_classes = 100
        assert args.nb_classes == nb_classes
    elif args.data_set == 'TINY_IMAGENET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        if not args.convert_to_ffcv :
            dataset = datasets.ImageFolder(root, transform=transform)
        else:
            dataset = datasets.ImageFolder(root)
        nb_classes = 200
        assert args.nb_classes == nb_classes
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        if not args.convert_to_ffcv:
            dataset = datasets.ImageFolder(root, transform=transform)
        else:
            dataset = datasets.ImageFolder(root)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    
    print(f"Dataset type : {args.data_path}")
    print("Reading from datapath", args.data_path)
    print("Number of classes =  %d" % nb_classes)

    return dataset, nb_classes

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:  
            t.append(
            transforms.Resize((args.input_size, args.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None and args.input_size==224: 
                args.crop_pct = 224 / 256
            elif args.crop_pct is None:
                args.crop_pct = 1.0
            size = int(args.input_size / args.crop_pct)
            t.append(
                transforms.Resize((size,size), interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    print(t)
    return transforms.Compose(t)


class KaggleISICDataset(Dataset):
    def __init__(self, csv_file, image_dir, skin_color_csv = None, transform=None, augment_transforms = None,split='train', test_size=0.2, seed=42):
        """
        Args:
            csv_file (str): Path to the CSV file containing image names and targets.
            image_dir (str): Directory containing the image files.
            skin_color_csv (str): Path to the CSV file containing information about each image ita, fitzpatrick scale and group it belongs
                - ita: The individual typology angle (ITA) is a measure of the skin's reaction to sun exposure. 
                - fitzpatrick_scale: The Fitzpatrick scale is a numerical classification schema for human skin color. 
                - group : Each patient is classified into one of the groups based on his skin color type.
            transform (callable, optional): Optional transform to be applied on an image.
            augment_transforms (dict): Dictionary containing augmentations to be applied on malignant cases.
            split (str): 'train' or 'valid', determines which subset to use.
            test_size (float): Proportion of the dataset to allocate for validation.
            seed (int): Random seed for reproducibility.
        """
        
        self.image_dir = image_dir
        self.transform = transform
        self.split = split
        
        if augment_transforms is None:
            self.augment_transforms = None
            self.oversample_ratio = 1
        else:
            self.augment_transforms = augment_transforms
            self.oversample_ratio = len(self.augment_transforms.keys())
            
        if skin_color_csv is not None:
            print(f"Skin color csv is defined. Using predefined train/valid splits.")
            df = pd.read_csv(skin_color_csv, sep=';') 
            train_df = df[df['split'] == "train"]
            valid_df = df[df['split'] == "train"]
        else:
            df = pd.read_csv(csv_file)     
            train_df, valid_df = train_test_split(
            df, test_size=test_size, stratify=df['target'], random_state=seed
        )
        
        
        self.data = train_df if split == 'train' else valid_df

        if skin_color_csv is not None:
            self.samples = [(row['image_name'], row['target'], row['group']) for _, row in self.data.iterrows()]
            self.groups =  len(self.data['group'].unique())
            self.group = self.data['group'].value_counts()
        else:
            self.samples = [(row['image_name'], row['target'], None ) for _, row in self.data.iterrows()]
            
        self.classes = 2  
        if self.augment_transforms is None:
            self.class_distribution = (len(self.data[self.data['target'] == 0]), len(self.data[self.data['target'] == 1]))
        else:
            self.class_distribution = (len(self.data[self.data['target'] == 0]), len(self.data[self.data['target'] == 1]) * self.oversample_ratio)

    def __len__(self):
        if self.augment_transforms is None or self.split != 'train':
            return len(self.data)
        return len(self.data[self.data['target'] == 0]) + self.oversample_ratio * len(self.data[self.data['target'] == 1])

    def __getitem__(self, idx):
        if self.augment_transforms is not None and self.split == 'train':
            actual_idx = min(idx // self.oversample_ratio, len(self.data) - 1) if idx % self.oversample_ratio != 0 else min(idx, len(self.data) - 1)
            row = self.data.iloc[actual_idx]
            augment_type = list(self.augment_transforms.keys())[idx % self.oversample_ratio]
        else:
            row = self.data.iloc[idx]
            augment_type = "original"

        image_name = row['image_name']
        if not image_name.lower().endswith('.jpg'):
            image_name = image_name + '.jpg'
            
        image_path = os.path.join(self.image_dir, image_name)
        target = row['target']  
        group = row['group'] if 'group' in row.index else -1
        image = Image.open(image_path).convert('RGB')

        if target == 1 and self.split == 'train' and self.augment_transforms is not None:
            image = self.augment_transforms[augment_type](image)
        if self.transform:
            image = self.transform(image)

        return image, target, group
    
class LocalISICDataset(Dataset):
    def __init__(self, root, transform = None, skin_color_csv = None, augment_transforms = None, split = 'train', cielab=False):
        """
        Args:
            root (str or ``pathlib.Path``): Root directory path.
            transform (callable, optional): A function/transform that takes in a PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            skin_color_csv (str): Path to the CSV file containing information about each image ita, fitzpatrick scale and group it belongs
                - ita: The individual typology angle (ITA) is a measure of the skin's reaction to sun exposure. 
                - fitzpatrick_scale: The Fitzpatrick scale is a numerical classification schema for human skin color. 
                - group : Each patient is classified into one of the groups based on his skin color type.
            augment_transforms (dict): Dictionary containing augmentations to be applied on malignant cases.
        """
        
        self.root = root
        self.transform = transform
        self.split = split
        
        if augment_transforms is None:
            self.augment_transforms = None
            self.oversample_ratio = 1  
        else:
            self.augment_transforms = augment_transforms
            self.oversample_ratio = len(self.augment_transforms.keys())
        
        split_dir = os.path.join(self.root, split)
        benign_dir = os.path.join(split_dir, 'benign')
        malignant_dir = os.path.join(split_dir, 'malignant')
        
        benign_images = [(os.path.join(benign_dir, img), 0) for img in os.listdir(benign_dir) 
                         if img.lower().endswith(('.jpg'))]
        malignant_images = [(os.path.join(malignant_dir, img), 1) for img in os.listdir(malignant_dir) 
                           if img.lower().endswith(('.jpg'))]
        self.samples = []
        self.samples.extend(benign_images)
        self.samples.extend(malignant_images)
        
        self.image_ids = [os.path.splitext(os.path.basename(path))[0] for path, _ in self.samples]
        
        self.benign_count = len(benign_images)
        self.malignant_count = len(malignant_images)
        self.classes = 2
        if self.augment_transforms is None:
            self.class_distribution = (self.benign_count, self.malignant_count)
        else:
            self.class_distribution = (self.benign_count, self.malignant_count * self.oversample_ratio)
        
        self.use_cielab = cielab

        if skin_color_csv is not None:
            self.skin_data = pd.read_csv(skin_color_csv, sep=';')
            self.samples_with_skin = []
            skin_info_dict = {}
            for _, row in self.skin_data.iterrows():
                img_name = row['image_name']
                if 'group' in self.skin_data.columns:
                    skin_info_dict[img_name] = row['group']
                    
            missing = 0
            for path, label in self.samples:
                img_filename = os.path.basename(path)
                
                if img_filename in skin_info_dict:
                    self.samples_with_skin.append((path, label, skin_info_dict[img_filename]))
                else:
                    missing+=1
    
            self.samples = self.samples_with_skin
            if skin_color_csv is not None:
                self.groups =  len(self.skin_data['group'].unique())
                self.group = self.skin_data['group'].value_counts()
    
    def __len__(self):
        if self.augment_transforms is None or self.split != 'train':
            return len(self.samples)
        if self.split == 'train':
            return self.benign_count + self.malignant_count * self.oversample_ratio
    
    def __getitem__(self, idx):
        if self.augment_transforms is not None and self.split == 'train' and idx >= self.benign_count:
            adjusted_idx = self.benign_count + ((idx - self.benign_count) // self.oversample_ratio)
            augment_type_idx = (idx - self.benign_count) % self.oversample_ratio
            augment_type = list(self.augment_transforms.keys())[augment_type_idx]
            
            if adjusted_idx >= len(self.samples):
                adjusted_idx = self.benign_count + ((adjusted_idx - self.benign_count) % self.malignant_count)
        else:
            adjusted_idx = idx
            augment_type = "original"
        
        if isinstance(self.samples[adjusted_idx], tuple) and len(self.samples[adjusted_idx]) == 3:
            image_path, target, group = self.samples[adjusted_idx]
        else:
            image_path, target = self.samples[adjusted_idx]
            group = -1
          
        if not self.use_cielab:
            image = Image.open(image_path).convert('RGB')
        else:
            image = Image.fromarray(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2LAB))

        if target == 1 and self.split == 'train' and augment_type != "original" and self.augment_transforms is not None:
            image = self.augment_transforms[augment_type](image)
            
        if self.transform:
            image = self.transform(image)
            
        return image, target, group

    def get_class_distribution(self):
        return self.class_distribution
       