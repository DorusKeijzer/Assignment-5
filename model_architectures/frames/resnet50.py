import torch
import torch.nn as nn
import torchvision.models as models

class model(nn.Module):
    name = "resnet50"  # change to reflect model version
    intermediate_layers = False

    def __init__(self):
        super(model, self).__init__()
        # Load pre-trained ResNet model
        resnet = models.resnet50(pretrained=True)
        
        # Modify the final layer
        resnet.fc = nn.Linear(resnet.fc.in_features, 12)
        
        # Assign the modified ResNet to self.resnet
        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)
