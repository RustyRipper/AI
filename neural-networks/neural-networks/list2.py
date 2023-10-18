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
