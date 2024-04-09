import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch

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
    training_losses = []
    val_losses = []
    val_accuracies = []
    training_accuracies = []
    best_val_loss = 1e10
    best_epoch = 0
    best_model = None
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

        training_losses.append(avg_train_loss)
        val_loss, val_accuracy = evaluate_model(model, criterion, val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model = model.state_dict()
        T_loss, T_acc = evaluate_model(model, criterion, train_loader)

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        training_accuracies.append(T_acc)
        training_losses.append(T_loss)
        print("—————————————————————————————————————")
        print(f'Validation Loss: {val_loss:>20.4f}\nValidation Accuracy: {val_accuracy:>16.4f}')
        print(f'Training Loss: {T_loss:>22.4f}\nTraining Accuracy: {T_acc:>18.4f}\n')
    
    print(f"Best epoch: {best_epoch}")
    return {"val_loss":val_losses, 
            "train_loss": training_losses, 
            "val_accuracies": val_accuracies, 
            "training_accuracy": training_accuracies
            }, best_epoch, best_model

    
if __name__ == "__main__":
    from sys import argv
    # train_model "model path" "dataset" 
    model_path = argv[1]
    dataset = argv[2]
    import importlib.util

    def load_python_file(file_path):
        spec = importlib.util.spec_from_file_location("model", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    model: torch.nn.Module = load_python_file(model_path).model()

    if dataset == "OF":
        from data.HMDB51.dataset import optical_flow_test_dataloader, optical_flow_train_dataloader
        test_data = optical_flow_test_dataloader
        training_data = optical_flow_train_dataloader
        storage_location = "optical_flow"
    if dataset == "HMDB_frames":
        from data.HMDB51.dataset import mid_frame_test_dataloader, mid_frame_train_dataloader
        test_data = mid_frame_test_dataloader
        training_data = mid_frame_train_dataloader
        storage_location = "frames/HMDB"

    if dataset == "stanford_frames":
        # TODO
        storage_location = "frames/HMDB"

    print(f"training {model.name} from {model_path} on {storage_location} data.")

    from torch import optim
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr= 0.001)

    trainingstats, best_epoch, best_model = train(model, training_data, test_data, criterion, optimizer)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    filename = f"{model.name}_epoch_{best_epoch}_{timestamp}.pth"
    print(f"Saving best model to {filename}")
    torch.save(best_model, f"trained_models/{storage_location}/{filename}")