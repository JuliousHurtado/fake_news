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
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

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
        if len(vects) > 0:
            return torch.stack(vects)
        return []

class ManageData(object):
    """docstring for ManageData"""
    def __init__(self, path):
        super(ManageData, self).__init__()
        self.path = path

        self.x1_train = []
        self.x2_train = []
        self.target_train = []

        self.x1_test = []
        self.x2_test = []
        self.target_test = []

        self.features = Features()
        
        self.test_final = self.readFile('test.csv')

        """
        self.train_c = self.readFile('train.csv')
        self.splitData()
        #self.countLabels()
        self.getVectors()
        self.save_data()
        """

        self.load_Data()

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

    def save_data(self):
        base_path = '/mnt/nas2/GrimaRepo/jahurtado/dataset/WSDM/'
        torch.save({ 'x1': self.x1_train, 'x2': self.x2_train, 'target': self.target_train }, base_path + 'train.pth.tar')
        torch.save({ 'x1': self.x1_test, 'x2': self.x2_test, 'target': self.target_test }, base_path + 'test.pth.tar')

    def load_Data(self):
        base_path = '/mnt/nas2/GrimaRepo/jahurtado/dataset/WSDM/'

        checkpoint = torch.load(base_path + 'train.pth.tar')
        self.x1_train = checkpoint['x1']
        self.x2_train = checkpoint['x2']
        self.target_train = checkpoint['target']

        checkpoint = torch.load(base_path + 'test.pth.tar')
        self.x1_test = checkpoint['x1']
        self.x2_test = checkpoint['x2']
        self.target_test = checkpoint['target']


    def getVectors(self):
        i = 0
        print("Loading Train")
        for elem1,elem2,target in zip(self.train.title1_en, self.train.title2_en, self.train.label): 
            vect1 = self.features.tokenizeText(elem1)
            vect2 = self.features.tokenizeText(elem2)

            self.x1_train.append(vect1)
            self.x2_train.append(vect2)

            self.target_train.append(self.defineTarget(target))

            printProgressBar(i, len(self.train), prefix = 'Progress:', suffix = 'Complete', length = 50)
            i += 1

        i = 0
        print("Loading Test")
        for elem1,elem2,target in zip(self.test.title1_en, self.test.title2_en, self.test.label): 
            vect1 = self.features.tokenizeText(elem1)
            vect2 = self.features.tokenizeText(elem2)

            self.x1_test.append(vect1)
            self.x2_test.append(vect2)

            self.target_test.append(self.defineTarget(target))

            printProgressBar(i, len(self.test), prefix = 'Progress:', suffix = 'Complete', length = 50)
            i += 1

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

    def getData(self, train = True):
        if train:
            for i,elem in enumerate(self.x1_train):

                if type(elem) == list:
                    elem = torch.zeros(300) 
                if type(self.x2_train[i]) == list:
                    self.x2_train[i] = torch.zeros(300) 

                yield elem, self.x2_train[i],self.target_train[i]
        else:
            for i,elem in enumerate(self.x1_test):
                if type(elem) == list:
                    elem = torch.zeros(300) 
                if type(self.x2_test[i]) == list:
                    self.x2_test[i] = torch.zeros(300) 
                    
                yield elem, self.x2_test[i],self.target_test[i]

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