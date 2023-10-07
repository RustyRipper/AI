import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
