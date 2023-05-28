from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data'
column_names = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
glass_data = pd.read_csv(data_url, names=column_names)

X = glass_data.drop('Type', axis=1)
y = glass_data['Type']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=53)


def processing(x_train, x_val):
    # None
    x_train_none = x_train
    x_val_none = x_val

    # Normalizer
    normalizer = Normalizer()
    x_train_norm = normalizer.fit_transform(x_train)
    x_val_norm = normalizer.transform(x_val)

    # StandardScaler
    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(x_train)
    x_val_std = scaler.transform(x_val)

    # PCA
    pca = PCA(n_components=2)
    x_train_pca = pca.fit_transform(x_train)
    x_val_pca = pca.transform(x_val)

    return [('None', x_train_none, x_val_none),
            ('Normalizer', x_train_norm, x_val_norm),
            ('StandardScaler', x_train_std, x_val_std),
            ('PCA', x_train_pca, x_val_pca)]


# NB classifiers
def nb_classifiers_create():
    nb_params = [
        1e-9,
        1e-8,
        1e-7
    ]
    nb_classifiers = []

    for param in nb_params:
        nb = GaussianNB(var_smoothing=param)
        nb_classifiers.append(nb)

    return nb_classifiers


# DT classifiers
def dt_classifiers_create():
    dt_params = [
        {'max_depth': 5, 'min_samples_split': 5, 'criterion': 'gini'},
        {'max_depth': 10, 'min_samples_split': 5, 'criterion': 'gini'},
        {'max_depth': 15, 'min_samples_split': 5, 'criterion': 'gini'}
    ]

    dt_classifiers = []
    for params in dt_params:
        dt = DecisionTreeClassifier(criterion=params.get('criterion'), max_depth=params.get('max_depth'),
                                    min_samples_split=params.get('min_samples_split'))
        dt_classifiers.append(dt)

    return dt_classifiers


def run(classifiers):
    results = []

    for name, X_train_processed, X_val_processed in processing(X_train, X_val):
        for i, classifier in enumerate(classifiers):
            row = [name + str(i)]
            classifier.fit(X_train_processed, y_train)
            y_pred = classifier.predict(X_val_processed)

            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, average='weighted')
            recall = recall_score(y_val, y_pred, average='weighted')
            f1 = f1_score(y_val, y_pred, average='weighted')

            row.extend([accuracy, precision, recall, f1])
            results.append(row)

            cm = confusion_matrix(y_val, y_pred)
            print("\nConfusion Matrix for {}: \n{}".format(name + str(i), cm))

    return results


def show_table(title, results):
    print("\n\n\n")
    print("Klasyfikator   " + title + "     | Dokladnosc Precyzja   Czulosc  F1-score  |")
    print("-" * 70)
    for row in results:
        print("{:20s}  |  {:.6f}  {:.6f}  {:.6f}  {:.6f}  |".format(row[0], row[1], row[2], row[3], row[4]))


def plot_results(results, nr):
    plt.figure(figsize=(12, 6))
    scores = [row[nr] for row in results]
    hyperparameters = [row[0] for row in results]
    plt.bar(hyperparameters, scores)
    plt.ylabel(str(nr))
    plt.xticks(rotation=45)
    plt.ylim(0.7, 1.0)
    plt.tight_layout()
    plt.show()


nb_results = run(nb_classifiers_create())
dt_results = run(dt_classifiers_create())

show_table('NB', nb_results)
show_table('DT', dt_results)

for i in range(1, 5):
    plot_results(nb_results, i)
    plot_results(dt_results, i)
