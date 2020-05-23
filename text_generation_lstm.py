# https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

import numpy as np
import nltk
from keras.models import Sequential
from keras.layers import LSTM,Dense
import os
import warnings
warnings.filterwarnings('ignore')

filename = "data.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()

print(raw_text)

#raw_text = ''.join(e for e in raw_text if e.isalnum())
raw_text = raw_text.replace('â','')
raw_text = raw_text.replace("œ","")
raw_text = raw_text.replace("å","")
raw_text = raw_text.replace("€","")
raw_text = raw_text.replace("€","")
raw_text = raw_text.replace("™","")
raw_text = raw_text.replace('\n',"")

# unique chars
chars = sorted(list(set(raw_text)))
#print(chars)

n_chars = len(raw_text)
n_vocab = len(chars)
# print("Total Characters: ", n_chars)
# print("Total Vocab: ", n_vocab)

dict = {}
for i in range(0,len(chars)):
    dict[chars[i]] = i


print(dict)


CHAR_LENGTH = 100
prev_chars = []
next_chars = []
for i in range(n_chars - CHAR_LENGTH):
    char_list = []
    for char in raw_text[i:i + CHAR_LENGTH]:
        char_list.append(char)
    prev_chars.append(char_list)
    next_chars.append(raw_text[i + CHAR_LENGTH])


# for i in range(0,len(prev_chars)):
#     print('previous: ')
#     print(prev_chars[i])
#     print('next: ')
#     print(next_chars[i])

X = np.zeros((len(prev_chars),CHAR_LENGTH,n_vocab),dtype=int)
y = np.zeros((len(next_chars),n_vocab),dtype=int)

for i in range(0,len(prev_chars)):
    for j in range(0, CHAR_LENGTH):
        X[i][j][dict[prev_chars[i][j]]] = 1

#print('X',X)

for i in range(0,len(next_chars)):
    for j in range(0, n_vocab):
        y[i][dict[next_chars[i]]] = 1


#print('y',y)


def createModel():
    # Building the model
    # We use a single-layer LSTM model with 128 neurons,
    #  a fully connected layer, and a softmax function for activation.

    model = Sequential()
    # X has shape (len(prev_words),WORD_LENGTH,len(unique_words))
    # so, sample shape(shape[0] will not go here
    # By default, return_sequences=False.
    # If we want to add more LSTM layers,
    # then the last LSTM layer must add return_sequences=True
    model.add(LSTM(128, input_shape=(CHAR_LENGTH, n_vocab)))

    # output class number, here it will be the unique words number
    model.add(Dense(n_vocab, activation='softmax'))

    from keras.optimizers import RMSprop
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(X, y, validation_split=0.05, batch_size=128, epochs=2, shuffle=True)

    # save this model
    model.save('keras_text_generation_model.h5')
    return model

def load_saved_model():
    from keras.models import load_model
    model = load_model('keras_text_generation_model.h5')
    return model


#model = createModel()
model = load_saved_model()

# input_text = input("input text of 100 characters")
input_text = "how she would feel with all their simple sorrows, and find a pleasure in all their simple joys, remembering her own child-life, and the happy summer days."
input_text = input_text.lower()

input_text = input_text[:100]
# print(len(input_text))
print('input Text:')
print(input_text)


input_text_list = [[c for c in input_text]]

X_test = np.zeros((len(input_text_list),CHAR_LENGTH,n_vocab),dtype=int)
#print(X_test,X_test.ndim)


for i in range(0,len(input_text_list)):
    for j in range(0, CHAR_LENGTH):
        X_test[i][j][dict[input_text_list[i][j]]] = 1

# print('X_test',X_test)
#
# print('prev: ')
# print(prev_chars)
# print('next')
# print(next_chars)

def generateChars(X_test,num_chars=100):

    generatedString = ""

    for i in range(0,num_chars):
        char = generateSingleCharacter(X_test)
        generatedString = generatedString + char

        # remove first character
        input_text_list[0].remove(input_text_list[0][0])

        # append the latest predicted character
        input_text_list[0].append(char)

        X_test = np.zeros((len(input_text_list), CHAR_LENGTH, n_vocab), dtype=int)

        for i in range(0, len(input_text_list)):
            for j in range(0, CHAR_LENGTH):
                X_test[i][j][dict[input_text_list[i][j]]] = 1

    print("Generated Text:")
    print(generatedString)


def generateSingleCharacter(X_test):
    y_pred = model.predict(X_test, verbose=0)[0]
    indices = np.argsort(y_pred)
    indices = indices[::-1]
    return chars[indices[0]]



generateChars(X_test)