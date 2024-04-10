import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, cnn_opticalflow, cnn_image, fusion_dim, num_classes):
        super(FusionModel, self).__init__()
        self.cnn_opticalflow = cnn_opticalflow
        self.cnn_image = cnn_image
        self.fc_fusion = nn.Linear(fusion_dim * 2, 512)  # Adjust the input dimension according to your concatenated feature size
        self.fc_output = nn.Linear(512, num_classes)  # Adjust the output dimension according to your task

    def forward(self, opticalflow, image):
        feature_opticalflow = self.cnn_opticalflow(opticalflow)
        feature_image = self.cnn_image(image)
        
        # Concatenate the features
        fused_features = torch.cat((feature_opticalflow, feature_image), dim=1)
        
        # Fusion layer
        fused_features = self.fc_fusion(fused_features)
        fused_features = torch.relu(fused_features)
        
        # Output layer
        output = self.fc_output(fused_features)
        return output
