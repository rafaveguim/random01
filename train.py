import numpy as np
import keras
import pickle

from keras.preprocessing.text import Tokenizer

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

from itertools import product

# Load the training data

training_data = np.genfromtxt(
    '/Volumes/Samsung_T5/data/random01/train.tsv',
    delimiter='\t', usecols=(0,1), dtype=None, comments=None)

# Decompose training data into two vectors
# x - input
# y - labels

train_x = [x[0] for x in training_data]
train_y = np.asarray([x[1] for x in training_data])

def bigram_generator(x):
    for i in range(2,len(x)+1):
        yield x[i-2:i]

# Build the vocabulary
# 1. Define observable letters
# 2. Define observable letter bigrams
# 3. Build lookup tables

letters = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~'

bigram2ind = dict()   # bigram -> index
vocab      = list()   # index  -> bigram
vocab_size = 0

for a, b in product(letters, repeat=2): # iterate in pairs
    bigram = bytes(a + b, 'utf-8')
    if bigram in bigram2ind:
        continue
    else:
        bigram2ind[bigram] = vocab_size
        vocab.append(bigram)
        vocab_size += 1

# Convert strings to index arrays (one-hot matrix)

matrix = np.zeros((len(train_x), vocab_size))
for i, x in enumerate(train_x):
    for bigram in bigram_generator(x):
        j = bigram2ind[bigram]
        matrix[i][j] = 1

train_x = matrix
train_y = keras.utils.to_categorical(train_y, 2)


model = Sequential()
model.add(Dense(512, input_shape=(vocab_size,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy'])

model.fit(train_x, train_y,
  batch_size=32,
  epochs=4,
  verbose=1,
  validation_split=0.1,
  shuffle=True)

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')


pickle.dump(vocab, open('vocab.pickle', 'wb'))
