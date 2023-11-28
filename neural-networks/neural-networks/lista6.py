import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import LazyLinear
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Load data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(1, 28, 28))
])


data = datasets.FashionMNIST('path', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST('path', train=False, download=True, transform=transform)

# Get x% Data
data_size_percent = 0.2
data_size = int(data_size_percent * len(data))
subset_data, _ = random_split(data, [data_size, len(data) - data_size])
test_data_size = int(data_size_percent * len(test_data))
subset_data_test, _ = random_split(test_data, [test_data_size, len(test_data) - test_data_size])

batch_size_default = 64

# DataLoader
train_loader = DataLoader(subset_data, batch_size=batch_size_default, shuffle=True)
test_loader = DataLoader(subset_data_test, batch_size=batch_size_default, shuffle=False)


def optimizer_criterion(model, learning_rate=0.05):
    return optim.Adagrad(model.parameters(), lr=learning_rate), nn.CrossEntropyLoss()


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
    std_devs_list = [0.14, 0.12, 0.1, 0.13]

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


class CNN(nn.Module):
    def __init__(self, output_channels, filter_size, pool_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=filter_size)
        self.conv2 = nn.Conv2d(16, output_channels, kernel_size=filter_size)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.flatten = nn.Flatten()
        self.linear = LazyLinear(100)
        self.output = nn.LogSoftmax(1)

        dummy_input = torch.randn(1, 1, 28, 28)
        _ = self.forward(dummy_input)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.linear(x)
        x = self.output(x)
        return x


output_channel_default = 32
filter_size_default = 3
pool_size_default = 2

# cnn_model = CNN(output_channel_default, filter_size_default, pool_size_default)
# optimizer, criterion = optimizer_criterion(cnn_model)
# train_model(cnn_model, train_loader, test_loader, criterion, optimizer)
#
# # Liczba kanałów wyjściowych warstwy konwolucyjnej
# output_channels = [16, 32]
#
# for output_channel in output_channels:
#     print(f"\noutput_channel: {output_channel}")
#     cnn_model = CNN(output_channel, filter_size_default, pool_size_default)
#     optimizer, criterion = optimizer_criterion(cnn_model)
#
#     train_model(cnn_model, train_loader, test_loader, criterion, optimizer)
#
# # 2 Rozmiar filtra warstwy konwolucyjnej
# filter_sizes = [3, 5]
#
# for filter_size in filter_sizes:
#     print(f"\nfilter_size: {filter_size}")
#
#     cnn_model = CNN(output_channel_default, filter_size, pool_size_default)
#     optimizer, criterion = optimizer_criterion(cnn_model)
#
#     train_model(cnn_model, train_loader, test_loader, criterion, optimizer)
#
# # 3 % train data
# pool_sizes = [2, 4]
#
# for pool_size in pool_sizes:
#     print(f"\npool_size: {pool_size}")
#
#     cnn_model = CNN(output_channel_default, filter_size_default, pool_size)
#     optimizer, criterion = optimizer_criterion(cnn_model)
#
#     train_model(cnn_model, train_loader, test_loader, criterion, optimizer)

# 4 gaussian noise
# Test
print(f"\n Gaussian TEST")
cnn_model = CNN(output_channel_default, filter_size_default, pool_size_default)
optimizer, criterion = optimizer_criterion(cnn_model)

train_model(cnn_model, train_loader, test_loader, criterion, optimizer, True)
# Test + Train

print(f"\n Gaussian TEST + TRAIN")
cnn_model = CNN(output_channel_default, filter_size_default, pool_size_default)
optimizer, criterion = optimizer_criterion(cnn_model)
train_model(cnn_model, train_loader, test_loader, criterion, optimizer, True, True)
