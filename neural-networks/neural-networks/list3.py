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
plt.figure(figsize=(12, 6))
for i, feature in enumerate(numeric_features.columns):
    plt.subplot(2, 3, i + 1)
    sns.histplot(data[feature], kde=True)
    plt.title(f'Rozkład {feature}')
plt.tight_layout()
plt.show()

# 4. Rozkład cech kategorycznych

categorical_features = data[['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']]
plt.figure(figsize=(12, 8))
for i, feature in enumerate(categorical_features.columns):
    plt.subplot(3, 3, i + 1)
    categorical_features[feature].value_counts().plot(kind='bar', title=f'Rozkład {feature}')
plt.tight_layout()
plt.show()

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
X = encoded_data.drop('num', axis=1)
y = encoded_data['num']
# Rozklad normalny
numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
scaler = preprocessing.StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

y = y.replace([1, 2, 3, 4], 1)
X = X.values
y = y.values
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X.shape)
print(y.shape)


class LogisticRegressionCustom:
    def __init__(self, learning_rate=0.01, num_iterations=1000, batch_size=32, random_seed=None, min_cost_diff=0.0001):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        self.min_cost_diff = min_cost_diff
        self.random_seed = random_seed
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

    @staticmethod
    def calculate_loss(y, y_predicted):
        return -1 * y * np.log(y_predicted) - (1 - y) * np.log(1 - y_predicted)

    @staticmethod
    def sigmoid(n):
        return 1 / (1 + np.exp(-n))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        print(self.weights.shape)

        cost_list = []
        for _ in range(self.num_iterations):

            if len(cost_list) > 1:
                cost_diff = abs(cost_list[-2] - cost_list[-1])
                print(cost_diff)
                if cost_diff < self.min_cost_diff:
                    break

            random = np.random.permutation(num_samples)

            X_random = X[random]
            y_random = y[random]

            for i in range(0, num_samples, self.batch_size):
                X_batch = X_random[i:i + self.batch_size]
                y_batch = y_random[i:i + self.batch_size]

                linear_model = np.dot(X_batch, self.weights) + self.bias
                y_predicted = self.sigmoid(linear_model)
                # shape 5,

                # X*Y = Z   Yt * Xt = Zt
                # (23,1)
                dw = (1 / self.batch_size) * np.dot(X_batch.T, (y_predicted - y_batch))
                # print(dw.shape)
                db = (1 / self.batch_size) * np.sum(y_predicted - y_batch)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

                cost = np.mean(self.calculate_loss(y_batch, y_predicted))
                cost_list.append(cost)

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_class)


# Inicjalizacja modelu regresji logistycznej
model = LogisticRegressionCustom(learning_rate=0.001, num_iterations=1000, batch_size=7, random_seed=77,
                                 min_cost_diff=0.001)

# Uczenie modelu
model.fit(X_train, y_train)

# Predykcja na zbiorze testowym
y_pred = model.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
# ===3==================================================================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)


#
# # Funkcje aktywacji
# def relu(x):
#     return np.maximum(0, x)
#
#
# def sigmoid(n):
#     return 1 / (1 + np.exp(-n))
#
#
# # Klasa warstwy ukrytej
# class HiddenLayer:
#     def __init__(self, input_dim_param, output_dim_param, activation):
#         self.output = None
#         self.input = None
#         self.weights = np.random.randn(input_dim_param, output_dim_param)
#         self.bias = np.zeros((1, output_dim))
#         self.activation = activation
#
#     def forward(self, x):
#         self.input = x
#         self.output = self.activation(np.dot(x, self.weights) + self.bias)
#         return self.output
#
#
# # Model sieci neuronowej
# class NeuralNetwork:
#     def __init__(self, input_dim_param, layers_param, learning_rate, batch_size):
#         self.layers = layers_param
#         self.learning_rate = learning_rate
#         self.batch_size = batch_size
#
#     @staticmethod
#     def calculate_loss(y, y_predicted):
#         return -1 * y * np.log(y_predicted) - (1 - y) * np.log(1 - y_predicted)
#
#     def forward(self, x):
#         for layer in self.layers:
#             x = layer.forward(x)
#         return x
#
#     def backward(self, x_batch, y_batch):
#         output = self.forward(x_batch)
#         loss = self.calculate_loss(y_batch, output)
#         for layer in reversed(self.layers):
#
#             d_output = y_batch - output
#             loss_hidden = d_output.dot(layer.weights.T)
#             d_hidden = loss_hidden * (layer.output > 0)
#             layer.weights += layer.input.T.dot(d_output) * self.learning_rate
#             layer.bias += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate
#             x_batch = d_hidden
#
#     def fit(self, X, y, num_iterations):
#         for epoch in range(num_iterations):
#             indices = np.arange(len(X))
#             np.random.shuffle(indices)
#             X_shuffled = X[indices]
#             y_shuffled = y[indices]
#             for i in range(0, len(X), self.batch_size):
#                 x_batch = X_shuffled[i:i + self.batch_size]
#                 y_batch = y_shuffled[i:i + self.batch_size]
#                 self.backward(x_batch, y_batch)
#
#     def predict(self, X):
#         y_predicted = []
#         for x in X:
#             y_predicted.append(self.forward(x))
#         y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
#         return np.array(y_predicted_class)
#
#
# # test
# input_dim = X_train.shape[1]
# hidden_dim = 1
# output_dim = 1
#
# # init
# layers = [
#     HiddenLayer(input_dim, hidden_dim, relu),
#     HiddenLayer(hidden_dim, output_dim, sigmoid)
# ]
#
# # init model
# model = NeuralNetwork(input_dim, layers, 0.01, 1)
#
# # Przeskaluj
# y_train = (y_train > 0).astype(int)
#
# # Uczenie modelu
# model.fit(X_train, y_train, 1000)
#
# # Testowanie modelu
# y_pred = model.predict(X_test)
#
# # Konwersja wyników na 0 lub 1
# print(y_test)
# print(y_pred)
# # Ocena modelu
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy}")


# ============================
class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate):
        self.output_layer_output = None
        self.output_layer_input = None
        self.hidden_layer_output = None
        self.hidden_layer_input = None
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.output_size = output_dim
        self.learning_rate = learning_rate

        # Init wagi i biasy
        self.weights_input_hidden = np.random.rand(input_dim, hidden_dim)
        self.bias_hidden = np.zeros((1, hidden_dim))
        self.weights_hidden_output = np.random.rand(hidden_dim, output_dim)
        self.bias_output = np.zeros((1, output_dim))

    def sigmoid(self, n):
        return 1 / (1 + np.exp(-n))

    def sigmoid_derivative(self, n):
        return n * (1 - n)

    def calculate_loss(self, y_pred, y):
        return - (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def forward(self, X):
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden

        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)

        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output

        self.output_layer_output = self.sigmoid(self.output_layer_input)

    def backward(self, X_batch, y_batch):
        #print(X_batch.shape)
        #print(y_batch.shape)
        #print(self.output_layer_output.shape)
        # Errors
        output_layer_error = y_batch - self.output_layer_output
        #print(output_layer_error.shape)

        output_layer_delta = output_layer_error * self.sigmoid_derivative(self.output_layer_output)
        #print(output_layer_delta.shape)

        hidden_layer_error = np.dot(output_layer_delta, self.weights_hidden_output.T)
        #print(hidden_layer_error.shape)
        hidden_layer_delta = hidden_layer_error * self.sigmoid_derivative(self.hidden_layer_output)

        # Update wagi i biasy
        self.weights_input_hidden += X_batch.T.dot(hidden_layer_delta) * self.learning_rate  # ( input -> hidden)

        self.weights_hidden_output += \
            (self.hidden_layer_output.T.dot(output_layer_delta) * self.learning_rate)  # (hidden -> output)

        self.bias_hidden += np.sum(hidden_layer_delta, axis=0, keepdims=True) * self.learning_rate

        self.bias_output += np.sum(output_layer_delta, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, num_iterations, batch_size):
        for iteration in range(num_iterations):
            total_error = 0
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                self.forward(X_batch)
                self.backward(X_batch, y_batch)

                batch_error = np.mean(self.calculate_loss(self.output_layer_output, y_batch))
                total_error += batch_error

            if (iteration + 1) % 100 == 0:
                print(f'iteration {iteration + 1}/{num_iterations}, Error: {total_error}')

    def predict(self, X):
        self.forward(X)
        return self.output_layer_output


print(X_train.shape)
print(y_train.shape)
nn = NeuralNetwork(input_dim=23, hidden_dim=14, output_dim=1, learning_rate=0.1)
nn.train(X_train, y_train, num_iterations=1000, batch_size=1)

y_pred = model.predict(X_test)
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
