import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch

class model(nn.Module):
    name = "three_convolution" #change to reflect model version
    intermediate_layers = False

    def __init__(self):
        super(model, self).__init__()
        # Define the convolutional layers
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Max pooling layer
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        # Fully connected layers
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 12)
        
        # dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Input size: [batch_size, channels, height, width]
        # Transpose the input tensor to [batch_size, channels, height, width]
        x = x.permute(0, 3, 1, 2)
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        
        # Flatten the output for fully connected layers
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.softmax(self.fc2(x), dim=0)
        
        return x
    

    
if __name__ == "__main__":
    pass