import torch
import torch.nn as nn
import torch.nn.init as init
import importlib
from model_architectures.frames.frames_resnet18 import model as frames_model
from model_architectures.optical_flow.four_layer import model as flow_model 


class model(nn.Module):
    name = "fusion_model"
    fusion_model = True
    def __init__(self):
        super(model, self).__init__()
        # pretrained models
        self.cnn_opticalflow = flow_model()
        self.cnn_image = frames_model()
        
        # 1 x 1 convulution for dimensionality reduction
        self.conv1x1_of = nn.Conv2d(in_channels=140, out_channels=64, kernel_size=1)  

        # fusion and classifier
        self.fc_fusion = nn.Linear(128, 256)  
        self.fc_hidden = nn.Linear(256, 256)  
        self.fc_output = nn.Linear(256, 12)  

        # batch norm layer
        self.bn1 = nn.BatchNorm1d(256)

        self.initialize_weights()

    def initialize_weights(self):
        of_weights = r"trained_models\optical_flow\four_layer_model_epoch_20_2024-04-13_16-56-22.pth"
        frames_weights = r"trained_models\frames\stanford\resnet18_custom_epoch_35_2024-04-11_12-17-20.pth"
        self.cnn_opticalflow.load_state_dict(torch.load(of_weights))# fill in path to model weights
        for param in self.cnn_opticalflow.parameters():
            param.requires_grad = False

        print(f"Succesfully loaded {of_weights}")

        self.cnn_image.load_state_dict(torch.load(frames_weights)) # fill in path to model weights
        for param in self.cnn_image.parameters():
            param.requires_grad = False

        print(f"Succesfully loaded {frames_weights}")

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.weight.requires_grad:
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None and m.bias.requires_grad:
                    init.constant_(m.bias, 0)


    def forward(self, input):
        input = input.permute(0, 3, 1, 2)

        # split optical flow from image
        optical_flow = input[:, 3:, :, :]  # Select the first 8 channels for optical flow

        image = input[:, :3, :, :]  # Select the remaining channels for image

        frame_prediction, frame_feature = self.cnn_image(image)
        frame_fused = torch.cat((frame_prediction, frame_feature), dim=1).unsqueeze(2).unsqueeze(3)

        # concatenate prediction and features
        of_prediction, of_feature = self.cnn_opticalflow(optical_flow)
        of_fused = torch.cat((of_prediction, of_feature), dim=1).unsqueeze(2).unsqueeze(3)
        # concatenate prediction and features

        # 1 x 1 convolution to reduce dimensionality 
        of_fused = self.conv1x1_of(of_fused)
        frame_fused = self.conv1x1_of(frame_fused)

        # Concatenate streams
        fused_features = torch.cat((of_fused, frame_fused), dim=1)

        fused_features = torch.flatten(fused_features, start_dim=1)
        # Fusion layer
        x = torch.relu(self.fc_fusion(fused_features))
        # batch norm    
        x = self.bn1(x)
        # hidden layers
        x = torch.relu(self.fc_hidden(x))

        x = torch.relu(self.fc_hidden(x))
        x = torch.relu(self.fc_hidden(x))

        # Output layer
        output = torch.softmax(self.fc_output(x), dim  =0)
        return output


if __name__ == "__main__":
    from torchsummary import summary
    model = model()
    batch_size = 8  # Define the batch size you want to use
    input_shape = (13, 224, 224)  # Define the input shape accordingly
    summary(model, input_shape)
