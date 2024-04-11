import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.init as init

class model(nn.Module):
    name = "frames_resnet18"  # change to reflect model version
    intermediate_layers = False

    def __init__(self):
        super(model, self).__init__()
        # Load pre-trained ResNet model
        resnet = models.resnet18(pretrained=True)
        
        # Modify the final layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc1 = nn.Linear(resnet.fc.in_features, 128)
        self.fc2 = nn.Linear(128, 12)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.weight.requires_grad:
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None and m.bias.requires_grad:
                    init.constant_(m.bias, 0)
   
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        penultimate_output = nn.ReLU()(self.fc1(x))
        output = nn.Softmax(dim=1)(self.fc2(penultimate_output))
        return output, penultimate_output
