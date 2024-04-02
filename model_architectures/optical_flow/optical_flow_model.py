import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class model(nn.Module):
    name = "Optical flow model" #change to reflect model version
    intermediate_layers = False

    def __init__(self, num_classes):
        raise NotImplementedError
    
    def forward():
        raise NotImplementedError
    
if __name__ == "__main__":
    from torchsummary import summary
    model = model(10)
    print(model.name)
    summary(model, (1, 28, 28))
