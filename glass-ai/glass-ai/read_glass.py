import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

column_names = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
data = pd.read_csv('glass (1).data', names=column_names)

print(data.head())

print(data.describe())

print(data['Type'].value_counts())

plt.figure(figsize=(8, 5))
data['Type'].value_counts().plot(kind='bar')
plt.xlabel('Type')
plt.ylabel('Count')
plt.title('rozklad klas')
plt.show()

correlation_matrix = data.drop('Id', axis=1).corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('macierz korelacji cech')
plt.show()

columns = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
plt.figure(figsize=(12, 10))
for i, feature in enumerate(columns, 1):
    plt.subplot(3, 3, i)
    for glass_type in data['Type'].unique():
        sns.histplot(data=data[data['Type'] == glass_type], x=feature, kde=True, label=f'Type {glass_type}')
    plt.xlabel(feature)
    plt.legend()
plt.tight_layout()
plt.show()
