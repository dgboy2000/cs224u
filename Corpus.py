# Corpus: for use in getting distributional information about text, such as unigrams/bigrams/trigrams

import nltk
from nltk.collocations import *
import DataSet
import LanguageUtils

CORPUS_CACHE = True

class Corpus:
    def __init__(self):
        self.corpus = list()

    def getWords(self):
        return self.corpus

    def setCorpus(self, type):
        if type == 'english-web':
            self.corpus = nltk.corpus.genesis.words('english-web.txt')
        if type == 'kaggle':
            self.corpus = self.getKaggleCorpus()
        else:
            raise Exception('Corpus does not exist.')

    def getKaggleCorpus(self):
        f = open('data/corpus.txt')
        text = f.read()
        tokens = LanguageUtils.tokenize(text)

        # TODO eventually:
        #tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        #tokenizer.tokenize(text.strip())

        return tokens
