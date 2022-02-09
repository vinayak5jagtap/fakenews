#Library initiation

import pandas as pd
import tensorflow as tf
import os
import re
import numpy as np
from string import punctuation
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GRU, LSTM, RNN, SpatialDropout1D

#Load Data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
train_data = train_data.set_index('id', drop = True)
test_data = test_data.set_index('id', drop = True)
min(train_data['length']), max(train_data['length']), round(sum(train_data['length'])/len(train_data['length']))
min(train_data['length']), max(train_data['length']), round(sum(train_data['length'])/len(train_data['length']))
max_features = 5000

#Tokenization of text
tokenizer = Tokenizer(num_words = max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True, split = ' ')
tokenizer.fit_on_texts(texts = train_data['text'])
X = tokenizer.texts_to_sequences(texts = train_data['text'])
tokenizer.fit_on_texts(texts = test_data['text'])
test_text = tokenizer.texts_to_sequences(texts = test_data['text'])

X = pad_sequences(sequences = X, maxlen = max_features, padding = 'pre')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)
test_text = pad_sequences(sequences = test_text, maxlen = max_features, padding = 'pre')

#LSTM Model selected for finding consistancy in the news data sequnce 
lstm_model = Sequential(name = 'lstm_nn_model')
lstm_model.add(layer = Embedding(input_dim = max_features, output_dim = 120, name = '1st_layer'))
lstm_model.add(layer = LSTM(units = 120, dropout = 0.2, recurrent_dropout = 0.2, name = '2nd_layer'))
lstm_model.add(layer = Dropout(rate = 0.5, name = '3rd_layer'))
lstm_model.add(layer = Dense(units = 120,  activation = 'relu', name = '4th_layer'))
lstm_model.add(layer = Dropout(rate = 0.5, name = '5th_layer'))
lstm_model.add(layer = Dense(units = len(set(y)),  activation = 'sigmoid', name = 'output_layer'))
lstm_model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#Model prediction and results saving
lstm_model_fit = lstm_model.fit(X_train, y_train, epochs = 10)
lstm_prediction = lstm_model.predict_classes(test_text)
output = pd.DataFrame({'id':test_data.index, 'label':lstm_prediction})
output.to_csv('Results.csv', index = False)
