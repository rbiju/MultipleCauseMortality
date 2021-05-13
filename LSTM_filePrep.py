import pandas as pd
import numpy as np
from tqdm import tqdm
import os

filepath = os.getcwd() + '/DIAGNOSES_ICD.csv'
df = pd.read_csv(filepath, header=0)

trainpath = os.getcwd() + '/trainfile.txt'
testpath = os.getcwd() + '/testfile.txt'
fTrain = open(trainpath, 'w')
fTest = open(testpath, 'w')


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
        for line in tqdm(lines):
            if line.strip("\n") != "nan":
                f.write(line.lower())
    f.close()


threshold = 0.8
patients = df.HADM_ID.unique()

# cutting down data to 10%
# patients = np.random.choice(patients, 10000, replace=False)

thresholdNdx = int(np.size(patients) * threshold)
patientsTrain = patients[0: thresholdNdx]
patientsTest = patients[thresholdNdx:]

print('Creating train file: \n')
createTextFile(fTrain, patientsTrain)

print('\nCreating test file: \n')
createTextFile(fTest, patientsTest)

print('Cleaning ...\n')
cleanFile(trainpath)
cleanFile(testpath)

print('Files ready.')
