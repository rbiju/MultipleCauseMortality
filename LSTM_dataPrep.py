import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
import pickle
import gensim
from gensim.models import Word2Vec
import os

# data obtained from MIMIC-III
# https://mimic.physionet.org/about/mimic/

filepath = os.getcwd() + '/DIAGNOSES_ICD.csv'
df = pd.read_csv(filepath, header=0)

textpath = os.getcwd() + '/textfile.txt'
trainpath = os.getcwd() + '/trainfile.txt'
testpath = os.getcwd() + '/testfile.txt'
fText = open(textpath, 'w')
fTrain = open(trainpath, 'w')
fTest = open(testpath, 'w')

splitRatio = 0.8
# number of sequences in 'memory' + 1
memConst = 4


class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self):
        corpus_path = os.getcwd() + '/textfile.txt'
        for line in open(corpus_path):
            yield line.split()


def makeICDDictionary(dataframe):
    code_column = dataframe.loc[:, 'ICD9_CODE']
    codes = code_column.values
    codes = np.ndarray.tolist(codes)

    seq = {}
    count = 1
    print('Making Library \n')
    for code in codes:
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

    # text file creation
    def npToSentence(npArr):
        funcSentence = ''.join([str(num) + ' ' for num in npArr])
        funcSentence = funcSentence.strip()

        return funcSentence + '\n'

    def createTextFile(f, patientList):
        for patient in tqdm(patientList):
            tempdf = df.loc[(df['HADM_ID'] == patient)]
            code_column = tempdf.loc[:, 'ICD9_CODE']
            codes = code_column.values
            sentence = npToSentence(codes)
            f.write(sentence)

        f.close()

    def cleanFile(path):
        with open(path, 'r') as f:
            lines = f.readlines()
        with open(path, 'w') as f:
            for line in lines:
                if line.strip("\n") != "nan":
                    f.write(line.lower())
        f.close()

    print('Creating Text Files.\n')
    createTextFile(fText, patients)
    createTextFile(fTrain, trainPatients)
    createTextFile(fTest, testPatients)
    cleanFile(textpath)
    cleanFile(trainpath)
    cleanFile(testpath)

    # word2vec model training
    print('\nTraining Word2Vec.\n')
    sentences = MyCorpus()
    w2v_model = gensim.models.Word2Vec(sentences=sentences, sg=1, min_count=1, vector_size=12)
    w2v_model.init_sims(replace=True)
    w2v_model.save("word2vec.model")
    print('\nWord2Vec Trained.\n')

    # sliding sequence creation
    def appendToMotherList(lowerMotherList, lowerDataFrame, lowerTrain_len):
        code_column = lowerDataFrame.loc[:, 'ICD9_CODE']
        codes = code_column.values
        patientCodeList = np.ndarray.tolist(codes)
        if len(patientCodeList) >= lowerTrain_len:
            for ndx in range(lowerTrain_len, len(patientCodeList)):
                minSeq = patientCodeList[ndx - lowerTrain_len: ndx]
                lowerMotherList.append(minSeq)
            return 0
        elif len(patientCodeList) < lowerTrain_len:
            return 1

    def makeSequences(patientArr, subList):
        rejectCount = 0
        for patient in tqdm(patientArr):
            tempdf = dataframe.loc[(df['HADM_ID'] == patient)]
            rejectCount += appendToMotherList(subList, tempdf, train_len)
        print('{} sequences were rejected'.format(rejectCount))

    print('Making Sequences.\n')
    makeSequences(patients, motherList)
    makeSequences(trainPatients, trainList)
    makeSequences(testPatients, testList)

    return motherList, trainList, testList


def makeVectorizedArray(textArray):
    w2v_model = Word2Vec.load("word2vec.model")
    dim = w2v_model.wv.vector_size
    narray = np.empty((np.shape(textArray)[0], np.shape(textArray)[1] * dim))
    for i, row in enumerate(textArray):
        vectorizedRow = np.array([])
        for j, element in enumerate(row):
            try:
                wordvec = w2v_model.wv[element.lower()]
                np.append(vectorizedRow, wordvec)
                vectorizedRow = np.concatenate((vectorizedRow, wordvec))
            except KeyError:
                wordvec = np.zeros(dim)
                print('{} not recognized'.format(element))
                vectorizedRow = np.concatenate((vectorizedRow, wordvec))
        narray[i] = vectorizedRow
    return narray


def sequenceToArray(textArray):
    text_data = np.array(textArray)[:, :-1]
    text_labels = np.array(textArray)[:, memConst - 1]
    text_labels = text_labels.reshape((np.shape(text_labels)[0], 1))
    labelArray = makeVectorizedArray(text_labels)
    dataArray = makeVectorizedArray(text_data)
    return dataArray, labelArray


text_sequences, train_sequences, test_sequences = makeMotherList(df, memConst, splitRatio)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

trainData, trainLabels = sequenceToArray(train_sequences)
testData, testLabels = sequenceToArray(test_sequences)

np.save(os.getcwd() + '/' + 'train_data.npy', trainData)
np.save(os.getcwd() + '/' + 'train_labels.npy', trainLabels)
np.save(os.getcwd() + '/' + 'test_data.npy', testData)
np.save(os.getcwd() + '/' + 'test_labels.npy', testLabels)

print('Process Complete')
