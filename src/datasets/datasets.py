import os
import random
from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

from .data_processing import ita_to_group

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
    # print(t)
    return transforms.Compose(t)


class KaggleISICDataset(Dataset):
    def __init__(self, csv_file, image_dir, skin_color_csv = None, transform=None, augment_transforms = None,split='train', test_size=0.2,seed=42):
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
    def __init__(self, root, transform = None, skin_color_csv = None, augment_transforms = None, split = 'train', cielab=False, skin_former = False, segment_out_skin = False):
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
            transform (callable, optional): Optional transform to be applied on an image.
            split (str): 'train' or 'valid', determines which subset to use.
            cielab (bool): If True, convert images to CIELAB color space.
            skin_former (bool): If True, use skin-former augmentation.
            segment_out_skin (bool): If True, segment out skin from the image.
        """
        
        self.root = root
        self.transform = transform
        self.split = split
        self.skin_fomer = skin_former
        self.segment_out_skin = segment_out_skin

        # Probability of applying skin transformations to dark
        self.group_shift_probs = [0.2, 0.4, 0, 0] 
        
        if augment_transforms is None:
            self.augment_transforms = None
            self.oversample_ratio = 1  
        else:
            self.augment_transforms = augment_transforms
            self.oversample_ratio = len(self.augment_transforms.keys())
        
        split_dir = os.path.join(self.root, split)
        benign_dir = os.path.join(split_dir, 'benign')
        malignant_dir = os.path.join(split_dir, 'malignant')
        
        benign_mask_dir = os.path.join(self.root, 'masks', split, 'benign')
        malignant_mask_dir = os.path.join(self.root, 'masks', split, 'malignant')
        
        benign_masks = [os.path.join(benign_mask_dir, mask) for mask in os.listdir(benign_mask_dir) 
                        if mask.lower().endswith(('.jpg'))]
        malignant_masks = [os.path.join(malignant_mask_dir, mask) for mask in os.listdir(malignant_mask_dir) 
                        if mask.lower().endswith(('.jpg'))]
        
        benign_images = [(os.path.join(benign_dir, img), 0) for img in os.listdir(benign_dir) 
                         if img.lower().endswith(('.jpg'))]
        malignant_images = [(os.path.join(malignant_dir, img), 1) for img in os.listdir(malignant_dir) 
                           if img.lower().endswith(('.jpg'))]
        
        benign_images = sorted(benign_images, key=lambda x: x[0])
        malignant_images = sorted(malignant_images, key=lambda x: x[0])
        benign_masks = sorted(benign_masks)
        malignant_masks = sorted(malignant_masks)
        
        self.samples = []
        self.samples.extend(zip(benign_images, benign_masks))
        self.samples.extend(zip(malignant_images, malignant_masks))
        
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
                    skin_info_dict[img_name] = {'ita': row['ita'], 'group': row['group']}
                    
            missing = 0
            for (path, label), mask in self.samples:
                img_filename = os.path.basename(path)
                
                if img_filename in skin_info_dict:
                    if not skin_former and not segment_out_skin:
                        self.samples_with_skin.append((path, label, skin_info_dict[img_filename]['group']))
                    else:
                        self.samples_with_skin.append((path, label, skin_info_dict[img_filename]['group'], skin_info_dict[img_filename]['ita'], mask))
                else:
                    missing+=1
    
            self.samples = self.samples_with_skin
            self.groups = 1
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
        
        if isinstance(self.samples[adjusted_idx], tuple) and len(self.samples[adjusted_idx]) == 3 and not self.skin_fomer:
            image_path, target, group = self.samples[adjusted_idx]
        elif isinstance(self.samples[adjusted_idx], tuple) and len(self.samples[adjusted_idx]) == 5 and (self.skin_fomer or self.segment_out_skin):
            image_path, target, group, ita, mask = self.samples[adjusted_idx]
        else:
            image_path, target = self.samples[adjusted_idx]
            group = -1
          
        if not self.use_cielab:
            image = Image.open(image_path).convert('RGB')
        elif (self.use_cielab and not self.skin_fomer and not self.segment_out_skin) or (self.split != 'train' and not self.segment_out_skin):
            image = Image.fromarray(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2LAB))
        elif self.use_cielab and self.skin_fomer:
            np_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2LAB)
            
            shift_prob = self.group_shift_probs[group]
            if random.random() < shift_prob:
                # Apply transformation
                mask = 1 - cv2.imread(mask)[:, :, 0] / 255
                assert len(mask.shape) == 2 or mask.shape[-1] == 1, "Mask has to be grayscale"
                assert mask.min() >= 0 and mask.max() <= 1, "Mask values have to be in [0,1]"

                # # Dummy mask, remove me!
                # if len(mask.shape) == 3:
                #     mask = np.zeros_like(np_image[:,:,0])
                #     mask[:224,:224] = 1

                # Max ita in our darkest groups (28)
                target_ita = random.random() * 38 - 10
                delta_ita = float(ita) - target_ita

                np_image[:, :, 2] += (mask * delta_ita * 0.5).astype(np.uint)   # Shift b
                np_image[:, :, 0] -= (mask * delta_ita * 0.12).astype(np.uint)  # Shift L

                # New label
                group = ita_to_group(target_ita)
            
            image = Image.fromarray(np_image)
        
        else:
            np_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2LAB)
            mask = cv2.imread(mask)[:, :, 0] / 255
            np_image = np_image.astype(np.float32)
            np_image *= mask[:, :, np.newaxis]
            np_image = np.clip(np_image, 0, 255).astype(np.uint8)
            
            image = Image.fromarray(np_image)
            
        if target == 1 and self.split == 'train' and augment_type != "original" and self.augment_transforms is not None:
            image = self.augment_transforms[augment_type](image)
            
        if self.transform:
            image = self.transform(image)
            
        return image, target, group

    def get_class_distribution(self):
        return self.class_distribution
    
    def get_labels(self):
        if self.augment_transforms is None or self.split != 'train':
            labels = []
            for sample in self.samples:
                if isinstance(sample, tuple) and len(sample) >= 2:
                    labels.append(sample[1])
                else:
                    img_data, _ = sample
                    _, target = img_data
                    labels.append(target)
            return labels
        else:
            labels = []
            for idx in range(len(self)):
                if idx < self.benign_count:
                    sample = self.samples[idx]
                else:
                    adjusted_idx = self.benign_count + ((idx - self.benign_count) // self.oversample_ratio)
                    if adjusted_idx >= len(self.samples):
                        adjusted_idx = self.benign_count + ((adjusted_idx - self.benign_count) % self.malignant_count)
                    sample = self.samples[adjusted_idx]
                
                if isinstance(sample, tuple) and len(sample) >= 2:
                    labels.append(sample[1])
                else:
                    img_data, _ = sample
                    _, target = img_data
                    labels.append(target)
            return labels