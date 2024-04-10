import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms.functional import resize
from PIL import Image

class HMDB51Dataset(Dataset):
    def __init__(self, annotations_file, image_dir, transform=None, target_transform=None, resize_shape=(240, 320)):
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
        image = resize(image, self.resize_shape)

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
        transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

    
mid_frame_test_data = HMDB51Dataset("data/HMDB51/mid_frame_test.csv", "mid_frames",val_data_transforms)
mid_frame_training_data = HMDB51Dataset("data/HMDB51/mid_frame_train.csv", "mid_frames", train_data_transforms)
mid_frame_val_data = HMDB51Dataset("data/HMDB51/mid_frame_val.csv", "mid_frames",val_data_transforms)

from torch.utils.data import DataLoader

mid_frame_train_dataloader = DataLoader(mid_frame_training_data, batch_size=1, shuffle=True,resize_shape=(244, 244))
mid_frame_test_dataloader = DataLoader(mid_frame_test_data, batch_size=1, shuffle=True,resize_shape=(244, 244))
mid_frame_val_dataloader = DataLoader(mid_frame_val_data, batch_size=1, shuffle=True,resize_shape=(244, 244))


optical_flow_test_data = HMDB51Dataset("data/HMDB51/of_test.csv", "optical_flow", resize_shape=(244, 244))
optical_flow_training_data = HMDB51Dataset("data/HMDB51/of_train.csv", "optical_flow", resize_shape=(244, 244))
optical_flow_val_data = HMDB51Dataset("data/HMDB51/of_val.csv", "optical_flow", resize_shape=(244, 244))

optical_flow_val_dataloader = DataLoader(optical_flow_val_data, batch_size=1, shuffle=True)
optical_flow_train_dataloader = DataLoader(optical_flow_training_data, batch_size=1, shuffle=True)
optical_flow_test_dataloader = DataLoader(optical_flow_test_data, batch_size=1, shuffle=True)
