import keras
import numpy as np
from keras import regularizers
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, LSTM, Dense
from keras.src.layers import SpatialDropout1D, Dropout
from keras.src.optimizers import Adam
from matplotlib import pyplot as plt

(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(
    path="imdb.npz",
    num_words=None,
    skip_top=0,
    maxlen=None,
    seed=113,
    start_char=1,
    oov_char=2,
    index_from=3
)
train_samples = int(0.2 * len(train_data))
test_samples = int(0.2 * len(test_data))

train_data = train_data[:train_samples]
train_labels = train_labels[:train_samples]

test_data = test_data[:test_samples]
test_labels = test_labels[:test_samples]

print(f"Train data size: {len(train_data)}")
print(f"Test data size: {len(test_data)}")

review_lengths = [len(review) for review in train_data]
avg_length = np.mean(review_lengths)
print(f"Avg length: {avg_length}")
max_length = np.max(review_lengths)
min_length = np.min(review_lengths)
print(f"Max length {max_length}")
print(f"Min length: {min_length}")

maxlen = 100
max_words = max(max(train_data[i]) for i in range(len(train_data))) + 1
print(f"Max words: {max_words}")
#max_length = 10
train_data = sequence.pad_sequences(train_data, maxlen=max_length)
test_data = sequence.pad_sequences(test_data, maxlen=max_length)


model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=64, input_length=max_length))
model.add(SimpleRNN(64))
model.add(Dense(1, activation='sigmoid'))

#optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels, epochs=5, batch_size=64, validation_split=0.2)

loss, accuracy = model.evaluate(test_data, test_labels)
print(f"Test accuracy: {accuracy}")


accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(accuracy) + 1), accuracy, label='Training Accuracy')
plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(loss) + 1), loss, label='Training Loss')
plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
