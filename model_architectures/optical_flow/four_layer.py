import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch

class model(nn.Module):
    name = "four_layer_model" #change to reflect model version
    intermediate_layers = False

    def __init__(self):
        super(model, self).__init__()
        # Define the convolutional layers
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=8,  out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16,  out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7)

        # pooling layers
        self.maxpool = nn.MaxPool2d((2,2),2)

        # Fully connected layers
        self.fc1 = nn.Linear(6400, 512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 12)
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
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool(x)
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)

        # Flatten the output for fully connected layers
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        penultimate = F.relu(self.fc3(x))
        out = F.softmax(self.fc4(penultimate), dim=1)
        
        return out, penultimate
    

    
if __name__ == "__main__":
    # from torchsummary import summary
    # model = model()
    # print(model.name)
    # summary(model, (8, 224, 224))  

    import pytorch_lightning as pl
    from torch.optim.lr_scheduler import OneCycleLR

    class Model(pl.LightningModule):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat, _ = self.model(x)
            loss = F.cross_entropy(y_hat, y)
            self.log('train_loss', loss)
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
            scheduler_dict = {
                "scheduler": OneCycleLR(optimizer, max_lr=0.1, epochs=self.trainer.max_epochs, steps_per_epoch=self.trainer.max_steps),
                "interval": "step"
            }
            return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

    class DataModule(pl.LightningDataModule):
        def __init__(self, train_dataloader, val_dataloader, test_dataloader):
            super().__init__()
            self.train_dataloader = train_dataloader
            self.val_dataloader = val_dataloader
            self.test_dataloader = test_dataloader

        def train_dataloader(self):
            return self.train_dataloader

        def val_dataloader(self):
            return self.val_dataloader

        def test_dataloader(self):
            return self.test_dataloader

    # Load your dataset loaders
    from data.HMDB51.dataset import optical_flow_test_dataloader, optical_flow_train_dataloader, optical_flow_val_dataloader

    # Create data module instance
    data_module = DataModule(optical_flow_train_dataloader, optical_flow_val_dataloader, optical_flow_test_dataloader)

    # Create model instance
    model_instance = Model(model)

    # Define trainer
    trainer = pl.Trainer(gpus=1) if torch.cuda.is_available() else pl.Trainer()

    # Use lr_find to find the best learning rate
    lr_finder = trainer.tuner.lr_find(model_instance, data_module)

    # Plot the lr_finder results
    fig = lr_finder.plot(suggest=True)
    fig.show()

    # Pick a learning rate based on the plot
    suggested_lr = lr_finder.suggestion()

    print("Suggested Learning Rate:", suggested_lr)
