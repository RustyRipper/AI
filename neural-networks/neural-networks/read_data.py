import pandas as pd
from bs4 import BeautifulSoup
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

ratings_data = pd.read_excel('jester_dataset_1_1/jester-data-1.xls', header=None)
ratings_data = ratings_data.iloc[:, 1:].replace(99, float('nan'))
ratings = ratings_data.mean()

jokes_data = []

for i in range(1, 101):
    file_name = f'jokes/init{i}.html'
    with open(file_name, 'r') as file:
        joke_html = file.read()
        soup = BeautifulSoup(joke_html, 'html.parser')
        joke_text = soup.find('font', size='+1').text.strip()
        jokes_data.append(joke_text)

model = SentenceTransformer('bert-base-cased')
embeddings = model.encode(jokes_data)

train_X, val_X, train_y, val_y = train_test_split(embeddings, ratings, test_size=0.2,
                                                  random_state=42)

print("Train X shape:", train_X.shape)
print("Train y shape:", train_y.shape)
print("Validation X shape:", val_X.shape)
print("Validation y shape:", val_y.shape)


def run(learning_rate_param=0.01, hidden_sizes=(100,), alpha_param=0.0):
    mlp = MLPRegressor(solver='sgd', alpha=alpha_param, learning_rate='constant',
                       learning_rate_init=learning_rate_param, hidden_layer_sizes=hidden_sizes)
    train_loss = []
    val_loss = []
    epochs = 700

    for epoch in range(epochs):
        mlp.partial_fit(train_X, train_y)

        pred_y = mlp.predict(train_X)
        train_loss.append(mean_squared_error(train_y, pred_y) / 2)
        pred_y = mlp.predict(val_X)
        val_loss.append(mean_squared_error(val_y, pred_y) / 2)

    # train_loss2 = mlp.loss_curve_
    # print(train_loss2)
    # print(train_loss)
    # print(val_loss)
    # plt.plot(range(len(train_loss2)), train_loss2, label=f'Train Loss2 {learning_rate_param} - {hidden_sizes}')
    plt.plot(range(len(train_loss)), train_loss, label=f'Train Loss {learning_rate_param} - {hidden_sizes} - {alpha_param}')
    plt.plot(range(len(val_loss)), val_loss, label=f'Validation Loss {learning_rate_param} - {hidden_sizes} - {alpha_param}')


# learning_rates = [0.001, 0.0001, 0.003]
#
# for lr in learning_rates:
#     run(learning_rate_param=lr)
#
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# hidden_layer_sizes = [(20,), (100,), (500,)]
#
# for size in hidden_layer_sizes:
#     run(0.0001, hidden_sizes=size)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()


alpha_list = [0.0, 0.0001, 0.00001]

for alpha in alpha_list:
    run(0.0001, hidden_sizes=(500,), alpha_param=alpha)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0.5, 1.2)
plt.legend()
plt.show()

for alpha in alpha_list:
    run(0.0001, hidden_sizes=(100,), alpha_param=alpha)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0.5, 1.2)
plt.legend()
plt.show()

for alpha in alpha_list:
    run(0.0001, hidden_sizes=(20,), alpha_param=alpha)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0.5, 1.2)
plt.legend()
plt.show()

# my_mlp = MLPRegressor(solver='sgd', alpha=0.0, learning_rate='constant',
#                       learning_rate_init=0.0001, hidden_layer_sizes=(20,), max_iter=400)
# # my_mlp = MLPRegressor()
#
# my_mlp.fit(train_X, train_y)
#
#
# def my_joke(joke):
#     joke_embedding = model.encode([joke])
#     joke_embedding = np.reshape(joke_embedding, (1, -1))
#     rating_prediction = my_mlp.predict(joke_embedding)
#     print("Predykcja oceny żartu:", rating_prediction)
#
#
# my_joke("What kind of dog does a magician have? A Labracadabrador!")
# my_joke("What's the best thing about Switzerland?, I don't know but the flag is a big plus")
# my_joke("The other day, my wife asked me to pass her lipstick, but I accidentally passed her a glue stick. She still "
#         "isn’t talking to me.")
# my_joke("My boss told me to have a good day. So I went home.")
