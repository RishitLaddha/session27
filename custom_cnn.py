"""
This module implements a custom CNN model for the MNIST dataset using only custom
layer functions defined in custom_layers.py. No high-level torch.nn layers are used
for the model's core operations.

This version uses a very small subset of the MNIST dataset for a quick test run.
"""

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from custom_layers import conv2d_custom, relu, max_pool2d_custom, flatten, linear_custom, softmax

# -----------------------------------------------------------------------------
# Custom CNN Model Class
# -----------------------------------------------------------------------------

class CustomCNN:
    def __init__(self):
        """
        Initializes the custom CNN model.
        Architecture:
          - Conv1: 1 input channel, 6 output channels, kernel size 5 (no padding)
          - ReLU activation
          - Max Pooling: 2x2 window, stride 2
          - Conv2: 6 input channels, 12 output channels, kernel size 5
          - ReLU activation
          - Max Pooling: 2x2 window, stride 2
          - Flatten and Fully Connected layer: maps from 12*4*4 features to 10 classes.
        """
        self.conv1_weight = torch.nn.Parameter(torch.randn(6, 1, 5, 5) * 0.1)
        self.conv1_bias   = torch.nn.Parameter(torch.zeros(6))
        self.conv2_weight = torch.nn.Parameter(torch.randn(12, 6, 5, 5) * 0.1)
        self.conv2_bias   = torch.nn.Parameter(torch.zeros(12))
        
        fc_in_features = 12 * 4 * 4
        fc_out_features = 10  # MNIST has 10 classes
        
        self.fc_weight = torch.nn.Parameter(torch.randn(fc_in_features, fc_out_features) * 0.1)
        self.fc_bias   = torch.nn.Parameter(torch.zeros(fc_out_features))
        
        self.params = [
            self.conv1_weight, self.conv1_bias,
            self.conv2_weight, self.conv2_bias,
            self.fc_weight, self.fc_bias
        ]
        self.device = torch.device("cpu")
    
    def to(self, device):
        self.device = device
        self.conv1_weight = self.conv1_weight.to(device)
        self.conv1_bias = self.conv1_bias.to(device)
        self.conv2_weight = self.conv2_weight.to(device)
        self.conv2_bias = self.conv2_bias.to(device)
        self.fc_weight = self.fc_weight.to(device)
        self.fc_bias = self.fc_bias.to(device)
        return self

    def forward(self, x):
        x = x.to(self.device)
        x = conv2d_custom(x, self.conv1_weight, self.conv1_bias, stride=1, padding=0)
        x = relu(x)
        x = max_pool2d_custom(x, kernel_size=2, stride=2)
        
        x = conv2d_custom(x, self.conv2_weight, self.conv2_bias, stride=1, padding=0)
        x = relu(x)
        x = max_pool2d_custom(x, kernel_size=2, stride=2)
        
        x = flatten(x)
        x = linear_custom(x, self.fc_weight, self.fc_bias)
        return x

    def get_parameters(self):
        return self.params

# -----------------------------------------------------------------------------
# Training and Evaluation Functions for the CNN
# -----------------------------------------------------------------------------

def train_cnn(model, train_loader, epochs=5, lr=0.01, device='cpu'):
    optimizer = optim.SGD(model.get_parameters(), lr=lr)
    logs = []
    total_samples = len(train_loader.dataset)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print("-" * 60)
        running_samples = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model.forward(images)
            # Compute cross-entropy loss manually:
            log_probs = torch.log(softmax(outputs, dim=1) + 1e-8)
            one_hot = torch.zeros_like(log_probs)
            one_hot.scatter_(1, labels.view(-1, 1), 1)
            loss = -torch.sum(one_hot * log_probs) / images.shape[0]
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            running_samples += images.shape[0]
            percent = (running_samples / total_samples) * 100
            print(f"Train Epoch: {epoch} [{running_samples}/{total_samples} ({percent:.0f}%)]\tLoss: {loss.item():.6f}")
        logs.append(f"Epoch {epoch+1}: Completed {running_samples}/{total_samples} samples")
    return logs

def evaluate_cnn(model, test_loader, device='cpu', epoch=None):
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model.forward(images)
        log_probs = torch.log(softmax(outputs, dim=1) + 1e-8)
        one_hot = torch.zeros_like(log_probs)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        loss = -torch.sum(one_hot * log_probs) / images.shape[0]
        
        preds = torch.argmax(outputs, dim=1)
        total_correct += torch.sum(preds == labels).item()
        total_loss += loss.item()
        total_samples += images.shape[0]
    
    avg_loss = total_loss / len(test_loader)
    accuracy = (total_correct / total_samples) * 100
    
    if epoch is not None:
        print(f"Epoch {epoch} Test set: Average loss: {avg_loss:.4f}, Accuracy: {total_correct}/{total_samples} ({accuracy:.2f}%)")
    else:
        print(f"Test set: Average loss: {avg_loss:.4f}, Accuracy: {total_correct}/{total_samples} ({accuracy:.2f}%)")
    return accuracy

# -----------------------------------------------------------------------------
# Main function for a Quick Test Run
# -----------------------------------------------------------------------------
def main():
    torch.manual_seed(42)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    full_train_dataset = torchvision.datasets.MNIST('./data', train=True, download=False, transform=transform)
    full_test_dataset = torchvision.datasets.MNIST('./data', train=False, download=False, transform=transform)
    
    # Using a small subset for quick testing:
    train_subset = Subset(full_train_dataset, range(200))  # adjust as needed
    test_subset = Subset(full_test_dataset, range(50))
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=50, shuffle=False)
    
    model = CustomCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print("Training Custom CNN on MNIST (training for 5 epochs)...")
    cnn_logs = train_cnn(model, train_loader, epochs=5, lr=0.01, device=str(device))
    evaluate_cnn(model, test_loader, device=str(device))
    
    with open("README_CNN.txt", "w") as f:
        f.write("Custom CNN Training Logs:\n")
        for log in cnn_logs:
            f.write(log + "\n")

if __name__ == "__main__":
    main()
