import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch

class optical_flow_model(nn.Module):
    name = "Optical flow model" #change to reflect model version
    intermediate_layers = False

    def __init__(self):
        super(optical_flow_model, self).__init__()
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
    
def evaluate_model(model, criterion, data_loader):
    loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            if model.intermediate_layers: # in case of multiple layers, average the loss over each layer
                for output in outputs:
                    loss += criterion(output, labels).item() * inputs.size(0)/len(outputs)
                # only the final output is used for prediction accuracy
                _, predicted = torch.max(outputs[0], 1)
            else:
                loss += criterion(outputs, labels).item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate average loss and accuracy
    avg_loss = loss / len(data_loader.dataset)
    accuracy = correct / total

    return avg_loss, accuracy

def train(net: nn.Module, train_loader, val_loader, criterion, optimizer: torch.optim.Adam, decrease_learning_rate = False):            #change last option to 'true' to implement decreasing learning rate for choice task 1
    print("Starting training...")
    training_loss = []
    eval_loss = []
    val_loss, val_accuracy = 0,0

    for epoch in range(1):
        print(f"Epoch {epoch+1}")
        print("=====================================")
        # if deacrease_learning_rate is set to True:
        if decrease_learning_rate and epoch > 0 and epoch % 5 == 0:
            # halves learning rate every 5th epoch for choice task #1
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                lr /= 2
                param_group['lr'] = lr 
            print(f"Halved learning rate to {lr}")

        train_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            # if the model implements intermediate layers, this will average the loss of each layer
            # if the model has only one layer, it works as normal
            outputs = model(inputs)
            if model.intermediate_layers:
                loss = 0
                for output in outputs:
                    loss += criterion(output, labels)/len(outputs)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

            if i % 100 == 0 or i == len(train_loader):  # Print when a batch is completed or it's the last batch
                avg_train_loss = train_loss / ((i + 1) * 1)
                print(f"Batch: {i:>3}/{len(train_loader)}, training loss: {avg_train_loss:.4f}")

        training_loss.append(avg_train_loss)
        val_loss, val_accuracy = evaluate_model(model, criterion, val_loader)
        T_loss, T_acc = evaluate_model(model, criterion, train_loader)

        eval_loss.append(val_loss)
        print("—————————————————————————————————————")
        print(f'Validation Loss: {val_loss:>20.4f}\nValidation Accuracy: {val_accuracy:>16.4f}')
        print(f'Training Loss: {T_loss:>20.4f}\nTraining Accuracy: {T_acc:>16.4f}\n')

    return eval_loss, training_loss, val_loss, val_accuracy 

    
if __name__ == "__main__":
    from dataset import optical_flow_test_dataloader, optical_flow_train_dataloader
    model = optical_flow_model()
    from torch import optim
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr= 0.001)

    a = train(model, optical_flow_train_dataloader, optical_flow_test_dataloader, criterion, optimizer)
    print(*a)