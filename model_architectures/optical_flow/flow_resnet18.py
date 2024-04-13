import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.init as init

class model(nn.Module):
    name = "flow_resnet18"
    intermediate_layers = False

    def __init__(self):
        super(model, self).__init__()
        # Load pre-trained ResNet model
        self.resnet = models.resnet18(pretrained=True)
        
        # Modify the first convolutional layer to accept 2-channel images
        self.resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify the final layer
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        self.fc1 = nn.Linear(self.resnet.fc.in_features, 128)
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
        # fixes the messed up order of input vector dimensions and data type
        x = x.permute(0, 3, 1, 2).float()

        x = self.features(x)
        x = torch.flatten(x, 1)
        penultimate_output = nn.ReLU()(self.fc1(x))
        output = nn.Softmax(dim=1)(self.fc2(penultimate_output))
        return output, penultimate_output

if __name__ == "__main__":
    from torchsummary import summary
    model = model()
    print(model.name)
    summary(model, (2, 224, 224))