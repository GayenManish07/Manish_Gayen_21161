import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#transforming data according to requirement
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


full_train_data = datasets.SVHN(root='./data', split='train', download=True, transform=data_transforms['train'])
full_test_data = datasets.SVHN(root='./data', split='test', download=True, transform=data_transforms['test'])

train_labels = full_train_data.labels
test_labels = full_test_data.labels

train_indices, _ = train_test_split(range(len(full_train_data)), test_size=0.25, stratify=train_labels)
test_indices, _ = train_test_split(range(len(full_test_data)), test_size=0.25, stratify=test_labels)

train_data = Subset(full_train_data, train_indices)
test_data = Subset(full_test_data, test_indices)


#data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4)

def train_model(model, criterion, optimizer, num_epochs=10):
    train_accuracy = []
    test_accuracy = []
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        corrects_train = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            corrects_train += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(train_data)
        epoch_acc_train = corrects_train.double() / len(train_data)
        train_accuracy.append(epoch_acc_train)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc_train:.4f}")

        # Evaluate on test set
        test_acc = evaluate_model(model)
        test_accuracy.append(test_acc)
        print(f"Epoch {epoch + 1}/{num_epochs}, Test Acc: {test_acc:.4f}")

    return train_accuracy, test_accuracy

# Evaluate model on the test set
def evaluate_model(model):
    model.eval()  
    corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
    test_acc = corrects.double() / len(test_data)
    return test_acc.item()



# ResNet-50 model
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # 10 output classes for SVHN
model = model.to(device)

# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_accuracy, test_accuracy = train_model(model, criterion, optimizer, num_epochs=10)


# Store accuracy in a text file
with open('ResNet50_accuracy.txt', 'w') as f:
    f.write("Epoch\tTrain Accuracy\tTest Accuracy\n")
    for epoch, (train_acc, test_acc) in enumerate(zip(train_accuracy, test_accuracy), 1):
        f.write(f"{epoch}\t{train_acc:.4f}\t{test_acc:.4f}\n")