'''
Token Classification - Most Basic Configuration
Using Vanilla LSTM Network Architecture

Author: Yuya Jeremy Ong (yuyajeremyong@gmail.com)
'''
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, TimeDistributed
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical

max_features = 20000
maxlen = 100  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

# Generate Arbitrary Binary Annotations - These labels are just for test, use actual data...
y_train = []
for x in X_train:
    y_train.append([[1] if i % 5 == 0 else [0] for i in x])

y_test = []
for x in X_test:
    y_test.append([[1] if i % 5 == 0 else [0] for i in x])

model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(LSTM(64, return_sequences=True))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.summary()

model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=4, validation_data=[X_test, y_test])
