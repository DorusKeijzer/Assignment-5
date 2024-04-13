import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch

class model(nn.Module):
    name = "optical_flow_models" #change to reflect model version
    intermediate_layers = False

    def __init__(self):
        super(model, self).__init__()
        # Define the convolutional layers
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=8,  out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        
        # pooling layers
        self.adaptivepool = nn.AdaptiveMaxPool2d((128, 128))
        self.maxpool = nn.MaxPool2d((2,2),2)

        # Fully connected layers
        self.fc1 = nn.Linear(4096, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 12)

        # dropout layer
        self.dropout = nn.Dropout(0.5)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.weight.requires_grad:
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None and m.bias.requires_grad:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # Input size: [batch_size, channels, height, width]
        # Transpose the input tensor to [batch_size, channels, height, width]
        x = x.permute(0, 3, 1, 2).float()
        # x = x.permute(1, 0, 3, 2)
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.adaptivepool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool(x)
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)

        # Flatten the output for fully connected layers
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        penultimate = F.relu(self.fc2(x))
        out = F.softmax(self.fc3(penultimate), dim=1)
        
        return out, penultimate
    

    
if __name__ == "__main__":
    from torchsummary import summary
    model = model()
    print(model.name)
    summary(model, (1, 224, 224))
