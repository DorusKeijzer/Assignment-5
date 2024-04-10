import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize
from PIL import Image

class StanfordDataSet(Dataset):
    def __init__(self, annotations_file, image_dir, transform=None, target_transform=None, resize_shape=(224, 224)):
        self.image_labels = pd.read_csv(annotations_file)
        self.image_dir = "data/stanford40/" + image_dir
        self.transform = transform
        self.target_transform = target_transform
        self.resize_shape = resize_shape
        # Get unique class labels and assign them as classes attribute
        self.classes = self.image_labels.iloc[:, 1].unique().tolist()

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
import torch

# mean and standard deviation for ImageNet normalization
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


train_data = StanfordDataSet("data/stanford40/test_annotation.csv", "frames",train_data_transforms)
val_data = StanfordDataSet("data/stanford40/val_annotation.csv", "frames",val_data_transforms)
test_data = StanfordDataSet("data/stanford40/train_annotation.csv", "frames",val_data_transforms)

from torch.utils.data import DataLoader

test_dataloader = DataLoader(test_data, batch_size=16, shuffle=False)
val_dataloader = DataLoader(val_data, batch_size=16, shuffle=False)
train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True  )


import torch
import torchvision
import matplotlib.pyplot as plt

def show_images_labels(batch, class_names):
    images, labels = batch

    # Create a grid of images
    num_images = len(images)
    num_rows = int(num_images / 4) + 1
    fig, axes = plt.subplots(num_rows, 4, figsize=(15, 4 * num_rows))

    for i, ax in enumerate(axes.flat):
        if i < num_images:
            image, label = images[i], labels[i]
            ax.imshow(image.permute(1, 2, 0))  # Convert from tensor (C, H, W) to numpy (H, W, C)
            ax.set_title(class_names[label])
            ax.axis('off')
        else:
            ax.axis('off')  # Remove axis if no image to display

    plt.show()

# Example usage
if __name__ == "__main__":

    class_names = train_data.classes

    for batch in train_dataloader:
        show_images_labels(batch, class_names)
        break  # Only display the first batch
