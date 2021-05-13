import os
import keras
import pickle
import numpy as np
from scipy.io import savemat

test_data = np.load(os.getcwd() + '/test_data.npy')
test_labels = np.load(os.getcwd() + '/test_labels.npy')
model = keras.models.load_model(os.getcwd() + '/MSP_NLP.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

preds = model.predict_classes(test_data)
test_ints = np.argmax(test_labels, axis=1)

embeddings = model.layers[0].get_weights()[0]

word_dict = {}
for i in range(1, np.shape(embeddings)[0]):
    word_dict[tokenizer.index_word[i]] = embeddings[i]

with open('word_dict.pickle', 'wb') as handle:
    pickle.dump(word_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

matdict = {'Predictions': preds, 'Labels': test_ints}
savemat('machine_predictions.mat', matdict)

print('done')
