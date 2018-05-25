import numpy as np
import keras
import pickle

from keras.models import model_from_json

vocab = pickle.load(open('vocab.pickle', 'rb'))
bigram2ind = dict(((bigram, i) for i, bigram in enumerate(vocab)))

with open('model.json', 'r') as model_file:
    model = model_from_json(model_file.read())
    model.load_weights('model.h5')

def bigram_generator(x):
    for i in range(2,len(x)+1):
        yield x[i-2:i]


def string2matrix(in_str):
    arr = np.zeros((1, len(vocab)))
    for bigram in bigram_generator(in_str):
        j = bigram2ind[bigram]
        arr[0][j] = 1
    return arr


def accuracy(test_set, true_labels):
    matrix = np.zeros((len(test_set), len(vocab)))
    for i, x in enumerate(test_set):
        for bigram in bigram_generator(x):
            j = bigram2ind[bigram]
            matrix[i][j] = 1

    M = model.predict(matrix)
    pred = np.argmax(M, 1)

    error = sum(pred ^ true_labels)

    return (len(test_set)-error)/len(test_set) # accuracy


def eval_list(strings):
    matrix = np.zeros((len(strings), len(vocab)))
    for i, x in enumerate(strings):
        for bigram in bigram_generator(bytes(x, 'utf-8')):
            j = bigram2ind[bigram]
            matrix[i][j] = 1

    M = model.predict(matrix)
    pred = np.argmax(M, 1)

    return zip(strings, pred)


def eval_string(string):
    labels = ['non-random', 'random']
    pred = model.predict(string2matrix(string))
    print("label: %s; confidence: %f%%" % (labels[np.argmax(pred)], pred[0][np.argmax(pred)] * 100))


# eval_string(b"R4F43L")

# Test

if __name__ == '__main__':
    test_data = np.genfromtxt(
        '/Volumes/Samsung_T5/data/random01/train.tsv',
        delimiter='\t', usecols=(0,1), dtype=None, comments=None)

    test_strings = [obs[0] for obs in test_data]
    true_labels  = [obs[1] for obs in test_data]

    print(accuracy(test_strings, true_labels))
