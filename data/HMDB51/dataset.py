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
        image = np.load(image_path)
        label = self.image_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    
mid_frame_test_data = HMDB51Dataset("data/HMDB51/mid_frame_test.csv", "mid_frames")
mid_frame_training_data = HMDB51Dataset("data/HMDB51/mid_frame_train.csv", "mid_frames")
mid_frame_val_data = HMDB51Dataset("data/HMDB51/mid_frame_val.csv", "mid_frames")

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
