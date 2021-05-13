import pandas as pd
from tqdm import tqdm
import os


filepath = os.getcwd() + '/DIAGNOSES_ICD.csv'
df = pd.read_csv(filepath, header=0)

f = open('SPADE_file.text', 'w')


def npToSentence(npArr):
    funcSentence = ''.join([str(num) + ' ' for num in npArr])
    funcSentence = funcSentence.strip()

    return funcSentence + '. '


patients = df.HADM_ID.unique()
print('Creating text file: \n')
for patient in tqdm(patients):
    tempdf = df.loc[(df['HADM_ID'] == patient)]
    code_column = tempdf.loc[:, 'ICD9_CODE']
    codes = code_column.values
    sentence = npToSentence(codes)
    f.write(sentence)

f.close()

print('File ready.')
