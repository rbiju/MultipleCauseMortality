import os
import keras
import pickle
import numpy as np
from gensim.models import Word2Vec
from scipy.io import savemat

memConst = 3
model = keras.models.load_model(os.getcwd() + '/MSP_NLP.h5')
w2v_model = Word2Vec.load("word2vec.model")
dim = w2v_model.wv.vector_size

test_data = np.load(os.getcwd() + '/test_data.npy')
test_data = test_data.reshape((np.shape(test_data)[0], memConst, dim))
test_labels = np.load(os.getcwd() + '/test_labels.npy')


model_predictions = model.predict(test_data)

rssults_lst = []
for ndx, row in enumerate(model_predictions):
    w2v_model.wv.similar_by_vector(row, topn=5, restrict_vocab=None)
print('done')
