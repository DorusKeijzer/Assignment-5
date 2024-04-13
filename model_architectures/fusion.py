import torch
import torch.nn as nn
import torch.nn.init as init
import importlib
from frames.frames_resnet18 import model as frames_resnet18
from optical_flow.flow_resnet18 import model as flow_resnet18


class model(nn.Module):
    name = "fusion_model"
    fusion_model = True
    def __init__(self):
        super(model, self).__init__()
        # pretrained models
        self.cnn_opticalflow = flow_resnet18()
        self.cnn_image = frames_resnet18()
        
        # 1 x 1 convulution for dimensionality reduction
        self.conv1x1_of = nn.Conv2d(in_channels=140, out_channels=64, kernel_size=1)  

        # fusion and classifier
        self.fc_fusion = nn.Linear(128, 256)  
        self.fc_hidden = nn.Linear(256, 256)  
        self.fc_output = nn.Linear(256, 12)  
        self.initialize_weights("", "")

    def initialize_weights(self, optical_flow_weights_path,frame_weights_path    ):
        if optical_flow_weights_path:
            # self.cnn_opticalflow.load_state_dict(torch.load(optical_flow_weights_path))
            for param in self.cnn_opticalflow.parameters():
                param.requires_grad = False

        if frame_weights_path:
            # self.cnn_image.load_state_dict(torch.load(frame_weights_path))
            for param in self.cnn_image.parameters():
                param.requires_grad = False
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.weight.requires_grad:
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None and m.bias.requires_grad:
                    init.constant_(m.bias, 0)


    def forward(self, input):
        # split optical flow from image
        optical_flow = input[:, :2, :, :]  # Select the first 2 channels for optical flow

        image = input[:, 2:, :, :]  # Select the remaining channels for image

        # concatenate prediction and features
        of_prediction, of_feature = self.cnn_opticalflow(optical_flow)
        of_fused = torch.cat((of_prediction, of_feature), dim=1).unsqueeze(2).unsqueeze(3)
        # concatenate prediction and features
        frame_prediction, frame_feature = self.cnn_image(image)
        frame_fused = torch.cat((frame_prediction, frame_feature), dim=1).unsqueeze(2).unsqueeze(3)

        # 1 x 1 convolution to reduce dimensionality 
        of_fused = self.conv1x1_of(of_fused)
        frame_fused = self.conv1x1_of(frame_fused)

        # Concatenate streams
        fused_features = torch.cat((of_fused, frame_fused), dim=1)

        fused_features = torch.flatten(fused_features, start_dim=1)
        # Fusion layer
        x = torch.relu(self.fc_fusion(fused_features))

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
    input_shape = (5, 244, 244)  # Define the input shape accordingly
    summary(model, input_shape)
