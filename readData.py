import pandas as pd 
import numpy as np 
import os
import spacy
import torch

from nltk.corpus import stopwords
import string

np.random.seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
id - the id of each news pair.
tid1 - the id of fake news title 1.
tid2 - the id of news title 2.
title1_zh - the fake news title 1 in Chinese.
title2_zh - the news title 2 in Chinese.
title1_en - the fake news title 1 in English.
title2_en - the news title 2 in English.
label - indicates the relation between the news pair: agreed/disagreed/unrelated.
"""
class Features(object):
    """docstring for Features"""
    def __init__(self):
        super(Features, self).__init__()
        self.nlp = spacy.load('en_core_web_lg')     

    # A custom function to tokenize the text using spaCy
    # and convert to lemmas
    def tokenizeText(self,sample):

        # get the tokens using spaCy
        tokens = self.nlp(sample.lower().strip())
        #print(tokens)
        # lemmatize
        vects = [ torch.from_numpy(tok.vector) for tok in tokens if not tok.is_stop ]
        return torch.stack(vects)

class ManageData(object):
    """docstring for ManageData"""
    def __init__(self, path):
        super(ManageData, self).__init__()
        self.path = path

        self.x1 = []
        self.x2 = []
        self.target = []

        self.features = Features()
        
        self.train_c = self.readFile('train.csv')
        self.test_final = self.readFile('test.csv')

        self.splitData()
        #self.countLabels()
        self.getVectors()

    def readFile(self, file):
        return pd.read_csv(os.path.join(self.path,file))

    def countLabels(self):
        print(self.train['label'].value_counts())
        print(self.test['label'].value_counts())

    def splitData(self):
        msk = np.random.rand(len(self.train_c)) < 0.8
        self.train = self.train_c[msk]
        self.test = self.train_c[~msk]

        print(len(self.train))
        print(len(self.test))

    def getVectors(self):
        for elem1,elem2,target in zip(self.train.title1_en, self.train.title2_en, self.train.label): 
            vect1 = self.features.tokenizeText(elem1)
            vect2 = self.features.tokenizeText(elem2)

            self.x1.append(vect1)
            self.x2.append(vect2)

            self.target.append(self.defineTarget(target))

        for elem1,elem2,target in zip(self.test.title1_en, self.test.title2_en, self.test.label): 
            vect1 = self.features.tokenizeText(elem1)
            vect2 = self.features.tokenizeText(elem2)

            self.x1.append(vect1)
            self.x2.append(vect2)

            self.target.append(self.defineTarget(target))

    def defineTarget(self, label):
        if label == 'agreed':
            return torch.tensor([0])
        elif label == 'disagreed':
            return torch.tensor([1])
        elif label == 'unrelated':
            return torch.tensor([2])
        else:
            print("Problema en los labels")
            print(label)

    def buildVocab(self):
        text = ''
        text2 = ''

        for elem1,elem2 in zip(self.train.title1_en, self.train.title2_en): 
            text += self.features.tokenizeText(elem1) + ' ' + self.features.tokenizeText(elem2) + ' '

        print(len(set(text.split())))
        #print(len(set(text2.split())))

    def getData(self):
        for i,elem in enumerate(self.x1):
            yield elem, self.x2[i],self.target[i]

    def getDataTrain(self):
        for elem1,elem2,target in zip(self.train.title1_en, self.train.title2_en, self.train.label): 
            vect1 = self.features.tokenizeText(elem1)
            vect2 = self.features.tokenizeText(elem2)

            y = self.defineTarget(target)

            yield vect1, vect2, y

    def getDataTest(self):
        for elem1,elem2,target in zip(self.test.title1_en, self.test.title2_en, self.test.label):
            vect1 = self.features.tokenizeText(elem1)
            vect2 = self.features.tokenizeText(elem2)

            y = self.defineTarget(target)

            yield vect1, vect2, y

    def getDataTestFinal(self):
        for elem1,elem2,t_id in zip(self.test_final.title1_en, self.test_final.title2_en, self.test_final.id):
            vect1 = self.features.tokenizeText(elem1)
            vect2 = self.features.tokenizeText(elem2)

            yield vect1, vect2, t_id

if __name__ == '__main__':
    ManageData("./data")