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

# Word2Vec model trained in LSTM_word2vec.py

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


def makeMotherList(dataframe, train_len, threshold):
    motherList = []
    patients = df.HADM_ID.unique()

    shufflePatients = patients
    thresholdNdx = int(np.size(patients) * threshold)
    np.random.shuffle(shufflePatients)
    trainPatients = shufflePatients[:thresholdNdx]
    trainList = []
    testPatients = shufflePatients[thresholdNdx:]
    testList = []

    # cutting down data to 50%
    # patients = np.random.choice(patients, 30000, replace=False)

    # takes in patient dataframe and append sliding sequences of length train_len to mother list
    # throws out sequence data whose length is < train_len
    def appendToMotherList(lowerMotherList, lowerDataFrame, HADM_ID, lowerTrain_len):
        code_column = lowerDataFrame.loc[:, 'ICD9_CODE']
        codes = code_column.values
        patientCodeList = np.ndarray.tolist(codes)
        for ndx in range(lowerTrain_len, len(patientCodeList)):
            if len(patientCodeList) > lowerTrain_len:
                minSeq = patientCodeList[ndx - lowerTrain_len: ndx]
                lowerMotherList.append(minSeq)
            else:
                print('HADM_ID {} had too few codes in sequence! (Less than {})'.format(HADM_ID, lowerTrain_len))

    def makeSequences(patientArr, list):
        for patient in tqdm(patientArr):
            tempdf = dataframe.loc[(df['HADM_ID'] == patient)]
            appendToMotherList(list, tempdf, patient, train_len)

    makeSequences(patients, motherList)
    makeSequences(trainPatients, trainList)
    makeSequences(testPatients, testList)

    return motherList, trainList, testList


def makeVectorizedArray(textArray):
    w2v_model = Word2Vec.load("word2vec.model")
    dim = w2v_model.wv.vector_size
    narray = np.empty((np.shape(textArray)[0], np.shape(textArray)[1] * dim))
    for i, row in tqdm(enumerate(textArray)):
        vectorizedRow = np.array([])
        for j, element in enumerate(row):
            try:
                wordvec = w2v_model.wv[element]
                np.append(vectorizedRow, wordvec)
                vectorizedRow = np.concatenate((vectorizedRow, wordvec))
            except KeyError:
                wordvec = np.zeros(dim)
                vectorizedRow = np.concatenate((vectorizedRow, wordvec))
        narray[i] = vectorizedRow
    return narray


def sequenceToArray(textArray):
    text_data = np.array(textArray)[:, :-1]
    text_labels = np.array(textArray)[:, memConst - 1]
    labelArray = np.array(tokenizer.texts_to_sequences(text_labels))
    dataArray = makeVectorizedArray(text_data)

    return dataArray, labelArray


text_sequences, train_sequences, test_sequences = makeMotherList(df, memConst, splitRatio)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

text_data = np.array(text_sequences)[:, :-1]
text_labels = np.array(text_sequences)[:, memConst-1]
tokenizedLabels = np.array(tokenizer.texts_to_sequences(text_labels))
nparray = makeVectorizedArray(text_data)

trainData = nparray[mask]
np.save(os.getcwd() + '/' + 'train_data.npy', trainData)
trainLabels = tokenizedLabels[mask]
# trainLabels = to_categorical(trainLabels, num_classes=vocabulary_size)
np.save(os.getcwd() + '/' + 'train_labels.npy', trainLabels)

testData = nparray[not mask]
np.save(os.getcwd() + '/' + 'test_data.npy', testData)
testLabels = tokenizedLabels[not mask]
# testLabels = to_categorical(testLabels, num_classes=vocabulary_size)
np.save(os.getcwd() + '/' + 'test_labels.npy', testLabels)

print('Process Complete')
