import torch
import torch.nn as nn
import torch.optim as optim

# architecture
class DenseNN(nn.Module):
    def __init__(self):
        super(DenseNN, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)  #  10 output classes

    def forward(self, x):
        x = torch.flatten(x, 1)  # Flattens the input images
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)  # softmax for multi-class classification
        return x

# Instantiate the model
model = DenseNN()

#loss function and optimizer
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters())


