import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Load data
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
data = datasets.FashionMNIST('path', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST('path', train=False, download=True, transform=transform)

# Get x% Data
data_size_percent = 1
data_size = int(data_size_percent * len(data))
subset_data, _ = random_split(data, [data_size, len(data) - data_size])
test_data_size = int(data_size_percent * len(test_data))
subset_data_test, _ = random_split(test_data, [test_data_size, len(test_data) - test_data_size])
#
# # Split Data
# train_size = int(0.8 * data_size)
# test_size = data_size - train_size
# train_data, test_data = random_split(subset_data, [train_size, test_size])

batch_size_default = 1000
# DataLoader
train_loader = DataLoader(subset_data, batch_size=batch_size_default, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size_default, shuffle=False)


def add_gaussian_noise(tensor, std=0.1):
    noise = torch.randn_like(tensor) * std
    return tensor + noise


def create_model(hidden_size_1=64, hidden_size2=None):
    if hidden_size2:
        return nn.Sequential(
            nn.Linear(28 * 28, hidden_size_1),
            nn.Sigmoid(),
            nn.Linear(hidden_size_1, hidden_size2),
            nn.Sigmoid(),
            nn.Linear(hidden_size2, 10),
            nn.Softmax(dim=1)
        )
    else:
        return nn.Sequential(
            nn.Linear(28 * 28, hidden_size_1),
            nn.Sigmoid(),
            nn.Linear(hidden_size_1, 10),
            nn.Softmax(dim=1)
        )


def optimizer_criterion(learning_rate=0.01):
    return optim.Adam(model.parameters(), lr=learning_rate), nn.CrossEntropyLoss()


def disturb_batch(input_batch, std):
    noise_batch = torch.randn_like(input_batch) * std
    disturbed_batch = input_batch + noise_batch
    return disturbed_batch


def train_model(model, train_loader, test_loader, criterion, optimizer, gauss_test=False, gauss_train=False,
                num_iterations=50):
    train_losses = []
    test_losses = []
    accuracies_train = []
    accuracies = []
    std_devs_list = [0.2, 0.15, 0.1, 0.15]

    for iteration in range(num_iterations):
        model.train()
        for inputs, labels in train_loader:
            if gauss_train:
                inputs = disturb_batch(inputs, std_devs_list[iteration % 4])

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Eval Train
        model.eval()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        with torch.no_grad():
            for inputs, labels in train_loader:
                if gauss_train:
                    inputs = disturb_batch(inputs, std_devs_list[iteration % 4])
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                train_loss += criterion(outputs, labels).item()

        # Eval test
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                if gauss_test:
                    inputs = disturb_batch(inputs, std_devs_list[iteration % 4])
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                test_loss += criterion(outputs, labels).item()

        # Save results
        train_losses.append(train_loss / len(train_loader))
        test_losses.append(test_loss / len(test_loader))
        train_accuracy = correct_train / total_train
        test_accuracy = correct_test / total_test
        accuracies_train.append(train_accuracy)
        accuracies.append(test_accuracy)

        print(
            f'Iteracja {iteration + 1}/{num_iterations}, Train Loss: {train_losses[-1]:.4f}, Test Loss:'
            f' {test_losses[-1]:.4f},'
            f'Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Iteracje')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(accuracies_train, label='Train Dokladnosc')
    plt.plot(accuracies, label='Test Dokladnosc')
    plt.xlabel('Iteracje')
    plt.ylabel('Dokladnosc')
    plt.legend()
    plt.show()


# 1 hidden size
# hidden_sizes = [64, 128]
#
# for hidden_size in hidden_sizes:
#     print(f"\nHidden Size: {hidden_size}")
#
#     model = create_model(hidden_size)
#     optimizer, criterion = optimizer_criterion()
#
#     train_model(model, train_loader, test_loader, criterion, optimizer)
#
# # 2 batch size
# batch_sizes = [16, 32]
#
# for batch_size in batch_sizes:
#     print(f"\nBatch Size: {batch_size}")
#
#     train_loader_2 = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#     test_loader_2 = DataLoader(test_data, batch_size=batch_size, shuffle=False)
#
#     model = create_model()
#     optimizer, criterion = optimizer_criterion()
#
#     train_model(model, train_loader_2, test_loader_2, criterion, optimizer)
#
# # 3 % train data
# training_sizes = [0.01, 0.1]
#
# for training_size in training_sizes:
#     print(f"\nTraining Size: {training_size}")
#
#     subset_train_size = int(training_size * train_size)
#     subset_train_data, _ = random_split(train_data, [subset_train_size, train_size - subset_train_size])
#     subset_train_loader = DataLoader(subset_train_data, batch_size=16, shuffle=True)
#
#     model = create_model()
#     optimizer, criterion = optimizer_criterion()
#     train_model(model, subset_train_loader, test_loader, criterion, optimizer)

# 4 gaussian noise
# Test
print(f"\n Gaussian TEST")
model = create_model()
optimizer, criterion = optimizer_criterion()

train_model(model, train_loader, test_loader, criterion, optimizer, True)
# Test + Train

print(f"\n Gaussian TEST + TRAIN")
model = create_model()
optimizer, criterion = optimizer_criterion()
train_model(model, train_loader, test_loader, criterion, optimizer, True, True)
