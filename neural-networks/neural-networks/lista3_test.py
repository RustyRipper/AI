import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('https://archive.ics.uci.edu/static/public/45/data.csv')
print(data.describe())

# 1. Zbalansowanie zbioru danych
class_counts = data['num'].value_counts()
print("\nLiczba próbek dla każdej klasy:")
print(class_counts)

# 2. Średnie i odchylenia cech liczbowych
numeric_features = data[['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']]

print("\nŚrednie cech liczbowych:")
mean_values = numeric_features.mean()
print(mean_values)
print("\nOdchylenia standardowe cech liczbowych:")
std_deviation = numeric_features.std()
print(std_deviation)

# 3. Rozkład cech liczbowych
# plt.figure(figsize=(12, 6))
# for i, feature in enumerate(numeric_features.columns):
#     plt.subplot(2, 3, i + 1)
#     sns.histplot(data[feature], kde=True)
#     plt.title(f'Rozkład {feature}')
# plt.tight_layout()
# plt.show()

# 4. Rozkład cech kategorycznych

categorical_features = data[['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']]
# plt.figure(figsize=(12, 8))
# for i, feature in enumerate(categorical_features.columns):
#     plt.subplot(3, 3, i + 1)
#     categorical_features[feature].value_counts().plot(kind='bar', title=f'Rozkład {feature}')
# plt.tight_layout()
# plt.show()

# 5. Cechy brakujące
missing_data = data.isnull().sum()
print("\nLiczba brakujących wartości dla każdej cechy:")
print(missing_data)

# Wypełnianie brakujących danych
mean = data['ca'].mean()
data['ca'].fillna(mean, inplace=True)

new_no_exist_category = 'none'
data['thal'].fillna(new_no_exist_category, inplace=True)

# Sprawdzenie czy zostało wypełnione
missing_data = data.isnull().sum()
print("\nLiczba brakujących wartości dla każdej cechy:")
print(missing_data)

# Przekształcenie cech kategorycznych za pomocą kodowania one-hot
encoded_data = pd.get_dummies(data, columns=['cp', 'restecg', 'slope', 'thal'],
                              prefix=['cp', 'restecg', 'slope', 'thal']).astype('int64')
print(encoded_data.columns)

# Po zastosowaniu kodowania one-hot cechy kategoryczne są teraz cechami liczbowymi
print("\nZakodowane cechy kategoryczne (one-hot encoding):")
print(encoded_data)
encoded_data.info()
print(encoded_data.describe())
print(encoded_data.head(303))


# 2######################################################################################
# 2######################################################################################
# 2######################################################################################

# Podział na dane wejściowe (X) i etykiety (y)
# X = encoded_data.drop('num', axis=1)
# y = encoded_data['num']
# # Rozklad normalny
# numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
# scaler = preprocessing.StandardScaler()
# X[numeric_features] = scaler.fit_transform(X[numeric_features])
#
# y = y.replace([1, 2, 3, 4], 1)
# X = X.values
# y = y.values
# print(X)
# print(y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X.shape)
# print(y.shape)
#
#
# class LogisticRegressionCustom:
#     def __init__(self, learning_rate=0.01, num_iterations=1000, batch_size=32, random_seed=None, min_cost_diff=0.0001):
#         self.learning_rate = learning_rate
#         self.num_iterations = num_iterations
#         self.batch_size = batch_size
#         self.weights = None
#         self.bias = None
#         self.min_cost_diff = min_cost_diff
#         self.random_seed = random_seed
#         if self.random_seed is not None:
#             np.random.seed(self.random_seed)
#
#     @staticmethod
#     def calculate_loss(y, y_predicted):
#         return -1 * y * np.log(y_predicted) - (1 - y) * np.log(1 - y_predicted)
#
#     @staticmethod
#     def sigmoid(n):
#         return 1 / (1 + np.exp(-n))
#
#     def fit(self, X, y):
#         num_samples, num_features = X.shape
#         self.weights = np.zeros(num_features)
#         self.bias = 0
#         print(self.weights.shape)
#
#         cost_list = []
#         for _ in range(self.num_iterations):
#
#             if len(cost_list) > 1:
#                 cost_diff = abs(cost_list[-2] - cost_list[-1])
#                 print(cost_diff)
#                 if cost_diff < self.min_cost_diff:
#                     break
#
#             random = np.random.permutation(num_samples)
#
#             X_random = X[random]
#             y_random = y[random]
#
#             for i in range(0, num_samples, self.batch_size):
#                 X_batch = X_random[i:i + self.batch_size]
#                 y_batch = y_random[i:i + self.batch_size]
#
#                 linear_model = np.dot(X_batch, self.weights) + self.bias
#                 y_predicted = self.sigmoid(linear_model)
#                 # shape 5,
#
#                 # X*Y = Z   Yt * Xt = Zt
#                 # (23,1)
#                 dw = (1 / self.batch_size) * np.dot(X_batch.T, (y_predicted - y_batch))
#                 # print(dw.shape)
#                 db = (1 / self.batch_size) * np.sum(y_predicted - y_batch)
#
#                 self.weights -= self.learning_rate * dw
#                 self.bias -= self.learning_rate * db
#
#                 cost = np.mean(self.calculate_loss(y_batch, y_predicted))
#                 cost_list.append(cost)
#
#     def predict(self, X):
#         linear_model = np.dot(X, self.weights) + self.bias
#         y_predicted = self.sigmoid(linear_model)
#         y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
#         return np.array(y_predicted_class)
#
#
# # Inicjalizacja modelu regresji logistycznej
# model = LogisticRegressionCustom(learning_rate=0.001, num_iterations=1000, batch_size=7, random_seed=77,
#                                  min_cost_diff=0.001)
#
# # Uczenie modelu
# model.fit(X_train, y_train)
#
# # Predykcja na zbiorze testowym
# y_pred = model.predict(X_test)
#
# # Ocena modelu
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
#
# print(f'Accuracy: {accuracy:.4f}')
# print(f'Precision: {precision:.4f}')
# print(f'Recall: {recall:.4f}')
# print(f'F1 Score: {f1:.4f}')
# ===3==================================================================================================================

class NeuralNetwork:
    def __init__(self, input_dim, output_dim, hidden_layers, learning_rate=0.01, weight_std=0.01, bias_std=0.01,
                 normalize_data=False):

        self.output_layer_output = None
        self.input_size = input_dim
        self.output_size = output_dim
        self.learning_rate = learning_rate
        self.weight_std = weight_std
        self.bias_std = bias_std
        self.normalize_data = normalize_data
        self.hidden_layers = hidden_layers
        self.layer_outputs = []

        self.weights = []
        self.biases = []

        layer_input_size = input_dim
        for layer_size in hidden_layers:
            self.weights.append(np.random.normal(0, weight_std, (layer_input_size, layer_size)))
            self.biases.append(np.random.normal(0, bias_std, (1, layer_size)))
            layer_input_size = layer_size

        self.weights.append(np.random.normal(0, weight_std, (layer_input_size, output_dim)))
        self.biases.append(np.random.normal(0, bias_std, (1, output_dim)))

    def sigmoid(self, n):
        return 1 / (1 + np.exp(-n))

    def sigmoid_derivative(self, n):
        return n * (1 - n)

    def normalize(self, X):
        return preprocessing.normalize(X)

    def forward(self, X):
        self.layer_outputs = []
        layer_output = X
        for i in range(len(self.weights)):
            layer_input = np.dot(layer_output, self.weights[i]) + self.biases[i]
            layer_output = self.sigmoid(layer_input)
            self.layer_outputs.append(layer_output)

        self.output_layer_output = layer_output

    def has_two_axes(self, arr):
        return len(arr.shape) >= 2

    def backward(self, X_batch, y_batch):
        output_layer_error = y_batch - self.output_layer_output.reshape(y_batch.shape)

        output_layer_delta = output_layer_error * self.sigmoid_derivative(self.layer_outputs[-1].reshape(y_batch.shape))
        output_layer_delta = output_layer_delta[:, np.newaxis]

        for i in range(len(self.weights) - 1, -1, -1):
            if self.has_two_axes(output_layer_delta):
                hidden_layer_error = np.dot(output_layer_delta, self.weights[i].T)
            else:
                hidden_layer_error = np.dot(output_layer_delta[:, np.newaxis], self.weights[i].T)

            # print(hidden_layer_error.shape)
            # print(self.sigmoid_derivative(self.layer_outputs[i]).shape)
            hidden_layer_delta = hidden_layer_error * self.sigmoid_derivative(self.layer_outputs[i-1])

            if i == 0:
                self.weights[i] += np.dot(X_batch.T, output_layer_delta) * self.learning_rate
            else:
                self.weights[i] += np.dot(self.layer_outputs[i-1].T, output_layer_delta) * self.learning_rate
            self.biases[i] += np.sum(output_layer_delta, axis=0, keepdims=True) * self.learning_rate

            output_layer_delta = hidden_layer_delta

    def fit(self, X, y, num_iterations, batch_size):
        if self.normalize_data:
            X = self.normalize(X)

        num_samples, num_features = X.shape
        cost_list = []

        for iteration in range(num_iterations):
            random = np.random.permutation(num_samples)
            X_random = X[random]
            y_random = y[random]
            total_error = 0

            for i in range(0, len(X), batch_size):
                X_batch = X_random[i:i + batch_size]
                y_batch = y_random[i:i + batch_size]

                self.forward(X_batch)
                self.backward(X_batch, y_batch)

                batch_error = np.mean(self.calculate_loss(self.output_layer_output, y_batch))
                total_error += batch_error

            avg_error = total_error / (len(X) / batch_size)
            cost_list.append(avg_error)

            if (iteration + 1) % 100 == 0:
                print(f'iteration {iteration + 1}/{num_iterations}, Avg Error: {avg_error}')

        return cost_list

    def calculate_loss(self, y_pred, y):
        return - (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def predict(self, X):
        if self.normalize_data:
            X = self.normalize(X)
        self.forward(X)
        return self.output_layer_output



X = encoded_data.drop('num', axis=1)
y = encoded_data['num']
# Rozklad normalny
numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
scaler = preprocessing.StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

y = y.replace([1, 2, 3, 4], 1)
X = X.values
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)

print(X_train.shape)
print(y_train.shape)
np.random.seed(7)
nn = NeuralNetwork(input_dim=23, hidden_layers=[15, 7, 4], output_dim=1, learning_rate=0.0011, weight_std=0.8,
                   bias_std=0.00001,
                   normalize_data=False)
nn.fit(X_train, y_train, num_iterations=1000, batch_size=10)

y_pred = nn.predict(X_test)
# print(y_pred)
y_pred = [1 if i > 0.5 else 0 for i in y_pred]
# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
