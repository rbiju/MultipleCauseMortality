import os
import numpy as np
import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Dense, LSTM, Dropout
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

# files generated by LSTM_dataPrep.py
train_data = np.load(os.getcwd() + '/train_data.npy')
train_labels = np.load(os.getcwd() + '/train_labels.npy')

test_data = np.load(os.getcwd() + '/test_data.npy')
test_labels = np.load(os.getcwd() + '/test_labels.npy')

w2v_model = Word2Vec.load("word2vec.model")
dim = w2v_model.wv.vector_size
memConst = 3
seq_len = train_data.shape[1]

# make data LSTM compatible
train_data = train_data.reshape((np.shape(train_data)[0], memConst, dim))
test_data = test_data.reshape((np.shape(test_data)[0], memConst, dim))

model = Sequential()

model.add(LSTM(64, input_shape=(memConst, dim), return_sequences=True))
model.add(LSTM(64))
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.1))
model.add(Dense(dim, activation='tanh'))

model.summary()

opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss=tf.keras.losses.CosineSimilarity(axis=-1), optimizer=opt)
history = model.fit(train_data, train_labels, batch_size=5000, epochs=75,
                    validation_data=(test_data, test_labels), verbose=True)

#  "Accuracy"
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('model_performance.png')

model.save('MSP_NLP.h5')
