import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms.functional import resize as tv_resize
from PIL import Image
import torch

class HMDB51Dataset_frame(Dataset):
    def __init__(self, annotations_file, image_dir, transform=None, target_transform=None, resize_shape=(224, 224)):
        self.image_labels = pd.read_csv(annotations_file)
        self.image_dir = "data/HMDB51/" + image_dir
        self.transform = transform
        self.target_transform = target_transform
        self.resize_shape = resize_shape

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_labels.iloc[idx, 0])
        image = Image.open(image_path).convert("RGB")  # Open image as PIL Image and convert to RGB
        label = self.image_labels.iloc[idx, 1]

        # Resize image to fixed size (with padding if necessary)
        image = tv_resize(image, self.resize_shape)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

from torchvision import transforms
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

val_data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Define the transform for training data with augmentation and ImageNet normalization
train_data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(200),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

    
mid_frame_test_data = HMDB51Dataset_frame("data/HMDB51/mid_frame_test.csv", "mid_frames",val_data_transforms)
mid_frame_training_data = HMDB51Dataset_frame("data/HMDB51/mid_frame_train.csv", "mid_frames", train_data_transforms)
mid_frame_val_data = HMDB51Dataset_frame("data/HMDB51/mid_frame_val.csv", "mid_frames",val_data_transforms)

from torch.utils.data import DataLoader

mid_frame_train_dataloader = DataLoader(mid_frame_training_data, batch_size=16, shuffle=True)
mid_frame_test_dataloader = DataLoader(mid_frame_test_data, batch_size=16, shuffle=True)
mid_frame_val_dataloader = DataLoader(mid_frame_val_data, batch_size=16, shuffle=True)

from skimage.transform import resize
from random import random
# Define custom transformation for two-channel numpy arrays
class TrainingTransforms(object):
    def __init__(self, resize_shape=(224, 224), mean=(0, 0), std=(1, 1)):
        self.resize_shape = resize_shape
        self.mean = mean
        self.std = std

    def __call__(self, optical_flow_field):
        # Resize optical flow field
        optical_flow_field = resize(optical_flow_field, self.resize_shape)

        # Random crop
        h, w = optical_flow_field.shape[:2]
        top = (h - self.resize_shape[0]) // 2
        left = (w - self.resize_shape[1]) // 2
        bottom = top + self.resize_shape[0]
        right = left + self.resize_shape[1]
        optical_flow_field = optical_flow_field[top:bottom, left:right]

        # random flip
        if random() < 0.5:  # 50% chance of flipping
            optical_flow_field = np.fliplr(optical_flow_field)

        # Normalize
        optical_flow_field = (optical_flow_field - self.mean) / self.std

        return optical_flow_field

class TestingTransforms(object):
    def __init__(self, resize_shape=(224, 224), mean=(0, 0), std=(1, 1)):
        self.resize_shape = resize_shape
        self.mean = mean
        self.std = std

    def __call__(self, optical_flow_field):
        # Resize optical flow field
        optical_flow_field = resize(optical_flow_field, self.resize_shape)

        # Normalize
        optical_flow_field = (optical_flow_field - self.mean) / self.std

        return optical_flow_field

class FusionTransforms(object):
    def __init__(self, resize_shape=(224, 224)):
        self.resize_shape = resize_shape

    def __call__(self, optical_flow_field):
        # Resize optical flow field
        optical_flow_field = resize(optical_flow_field, self.resize_shape)

        return optical_flow_field

class FusionTrainTransforms(object):
    def __init__(self, resize_shape=(224, 224)):
        self.resize_shape = resize_shape

    def __call__(self, optical_flow_field):
        if random() < 0.5:  # 50% chance of flipping
            optical_flow_field = np.fliplr(optical_flow_field)

        # Resize optical flow field
        optical_flow_field = resize(optical_flow_field, self.resize_shape)

        return optical_flow_field

class OF_data(Dataset):
    def __init__(self, annotations_file, optical_flow_field_dir, transform=None, target_transform=None, resize_shape=(240, 320)):
        self.optical_flow_field_labels = pd.read_csv(annotations_file)
        self.optical_flow_field_dir = "data/HMDB51/" + optical_flow_field_dir
        self.transform = transform
        self.target_transform = target_transform
        self.resize_shape = resize_shape

    def __len__(self):
        return len(self.optical_flow_field_labels)

    def __getitem__(self, idx):
        optical_flow_field_path = os.path.join(self.optical_flow_field_dir, self.optical_flow_field_labels.iloc[idx, 0])
        optical_flow_field = np.load(optical_flow_field_path)

        label = self.optical_flow_field_labels.iloc[idx, 1]

        if self.transform:
            optical_flow_field = self.transform(optical_flow_field)
        if self.target_transform:
            label = self.target_transform(label)
        return optical_flow_field, label


# Define transformations for training and validation data
resize_shape = (224, 224)
mean = (0, 0, 0, 0, 0, 0, 0, 0,)
std = (1, 1, 1, 1, 1, 1, 1, 1)

train_data_transforms = TrainingTransforms(resize_shape=resize_shape, mean=mean, std=std)
val_data_transforms = TestingTransforms(resize_shape=resize_shape, mean=mean, std=std)

# Create datasets and data loaders
optical_flow_test_data = OF_data("data/HMDB51/of_test.csv", "of_stacks", val_data_transforms)
optical_flow_training_data = OF_data("data/HMDB51/of_train.csv", "of_stacks", train_data_transforms)
optical_flow_val_data = OF_data("data/HMDB51/of_val.csv", "of_stacks", val_data_transforms)

optical_flow_val_dataloader = DataLoader(optical_flow_val_data, batch_size=16, shuffle=True)
optical_flow_train_dataloader = DataLoader(optical_flow_training_data, batch_size=16, shuffle=True)
optical_flow_test_dataloader = DataLoader(optical_flow_test_data, batch_size=16, shuffle=True)

fusion_transforms = FusionTransforms()
fursion_train_transforms = FusionTrainTransforms()

fusion_val_data = OF_data("data/HMDB51/of_val.csv", "fusion", fusion_transforms)
fusion_training_data = OF_data("data/HMDB51/of_train.csv", "fusion", fursion_train_transforms)
fusion_test_data = OF_data("data/HMDB51/of_test.csv", "fusion", fusion_transforms)

fusion_val_dataloader = DataLoader(fusion_val_data, batch_size=16, shuffle=True)
fusion_train_dataloader = DataLoader(fusion_training_data, batch_size=16, shuffle=True)
fusion_test_dataloader = DataLoader(fusion_test_data, batch_size=16, shuffle=True)
