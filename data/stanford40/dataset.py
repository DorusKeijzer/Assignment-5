import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms.functional import resize
from PIL import Image

class StanfordDataSet(Dataset):
    def __init__(self, annotations_file, image_dir, transform=None, target_transform=None, resize_shape=(240, 320)):
        self.image_labels = pd.read_csv(annotations_file)
        self.image_dir = "data/stanford40/" + image_dir
        self.transform = transform
        self.target_transform = target_transform
        self.resize_shape = resize_shape

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_labels.iloc[idx, 0])
        image = read_image(image_path)
        label = self.image_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    
test_data = StanfordDataSet("data/stanford40/test_annotation.csv", "frames")
val_data = StanfordDataSet("data/stanford40/val_annotation.csv", "frames")
train_data = StanfordDataSet("data/stanford40/train_annotation.csv", "frames")

from torch.utils.data import DataLoader

test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True)
train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)

