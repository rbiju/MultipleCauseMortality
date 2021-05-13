import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
import pickle
from gensim.models import Word2Vec
import os

# data obtained from MIMIC-III
# https://mimic.physionet.org/about/mimic/

filepath = os.getcwd() + '/DIAGNOSES_ICD.csv'
df = pd.read_csv(filepath, header=0)

splitRatio = 0.8
# number of sequences in 'memory' + 1
memConst = 4


def makeICDDictionary(dataframe):
    code_column = dataframe.loc[:, 'ICD9_CODE']
    codes = code_column.values
    codes = np.ndarray.tolist(codes)

    seq = {}
    count = 1
    print('Making Library \n')
    for code in tqdm(codes):
        if code not in seq:
            seq[code] = count
        count += 1
    print('\nLibrary Completed \n')
    return seq


def makeMotherList(dataframe, train_len):
    motherList = []
    patients = df.HADM_ID.unique()

    # cutting down data to 10%
    # patients = np.random.choice(patients, 30000, replace=False)

    # takes in patient dataframe and append sliding sequences of length train_len to mother list
    # throws out sequence data whose length is < train_len
    def appendToMotherList(motherList, dataframe, HADM_ID, train_len):
        code_column = dataframe.loc[:, 'ICD9_CODE']
        codes = code_column.values
        patientCodeList = np.ndarray.tolist(codes)
        for ndx in range(train_len, len(patientCodeList)):
            if len(patientCodeList) > train_len:
                minSeq = patientCodeList[ndx - train_len: ndx]
                motherList.append(minSeq)
            else:
                print('HADM_ID {} had too few codes in sequence! (Less than {})'.format(HADM_ID, train_len))

    print('Making Sequence List.\n')
    for patient in tqdm(patients):
        tempdf = dataframe.loc[(df['HADM_ID'] == patient)]
        appendToMotherList(motherList, tempdf, patient, train_len)
    print('Sequence List complete.\n')
    return motherList


def trainSplit(threshold, n_seq):
    thresholdNdx = int(np.shape(n_seq)[0] * threshold)
    train = n_seq[0: thresholdNdx, :]
    test = n_seq[thresholdNdx:, :]

    return train, test


text_sequences = makeMotherList(df, memConst)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

sequences = tokenizer.texts_to_sequences(text_sequences)

vocabulary_size = len(tokenizer.word_counts) + 1
n_sequences = np.empty([len(sequences), memConst], dtype='int32')
for i in range(len(sequences)):
    n_sequences[i] = sequences[i]

patientTrain, patientTest = trainSplit(splitRatio, n_sequences)

trainData = patientTrain[:, :-1]
np.save(os.getcwd() + '/' + 'train_data.npy', trainData)
trainLabels = patientTrain[:, -1]
# trainLabels = to_categorical(trainLabels, num_classes=vocabulary_size)
np.save(os.getcwd() + '/' + 'train_labels.npy', trainLabels)

testData = patientTest[:, :-1]
np.save(os.getcwd() + '/' + 'test_data.npy', testData)
testLabels = patientTest[:, -1]
# testLabels = to_categorical(testLabels, num_classes=vocabulary_size)
np.save(os.getcwd() + '/' + 'test_labels.npy', testLabels)

print('Process Complete')
