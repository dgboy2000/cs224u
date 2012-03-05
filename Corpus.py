# Corpus: for use in getting distributional information about text, such as unigrams/bigrams/trigrams

import nltk
from nltk.collocations import *
import DataSet
import LanguageUtils
import logging, gensim, bz2
import cPickle as pickle
from collections import Counter
import params

if params.DEBUG:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

CORPUS_CACHE = True

class Corpus:
    def __init__(self):
        self.corpus = list()
        self.pos_corpus = list()
        self.lsi = None

    def getWords(self):
        return self.corpus

    def getPOS(self):
        return self.pos_corpus

    def setCorpus(self, type, ds=None, ds2=None):
        if type == 'english-web':
            self.corpus = nltk.corpus.genesis.words('english-web.txt')
        elif type == 'ds':
            self.getFromDataSet(ds, ds2)
        else:
            raise Exception('Corpus does not exist.')

    def getFromDataSet(self, ds, ds2=None):
        c = list()
        for line in ds.getRawText():
            c += LanguageUtils.tokenize(line)
            c += ('', '', '')

        if ds2:
            for line in ds2.getRawText():
                c += LanguageUtils.tokenize(line)
                c += ('', '', '')

        self.corpus = c

        pos_corpus = list()
        for line in ds.getPOS():
            pos_corpus += line

        if ds2:
            for line in ds2.getPOS():
                pos_corpus += line

        self.pos_corpus = pos_corpus

    def genLSA(self, ds1, ds2):
        documents = list()
        for line in ds1.getRawText():
            documents.append(LanguageUtils.tokenize(line))
        for line in ds2.getRawText():
            documents.append(LanguageUtils.tokenize(line))

        # Remove stop words
        i = 0
        size = len(documents)
        texts = list()

        # You'd be amazed at how much faster dictionaries are over lists :)
        stop_words = nltk.corpus.stopwords.words('english')
        stop_dict = dict()
        for word in stop_words:
            stop_dict[word] = 1

        for doc in documents:
            texts.append([word for word in doc if word not in stop_dict])
            i += 1

        # Remove words that appear only once (TODO - maybe different min freq?)
        counts = Counter(self.getWords())

        i = 0
        size = len(texts)
        new_texts = list()
        for text in texts:
            new_texts.append([word for word in text if counts[word] > 1])
            i += 1
        texts = new_texts

        # create overall corpus
        dictionary = gensim.corpora.Dictionary(texts)
        #dictionary.save(TODO)

        mm_corpus = [dictionary.doc2bow(text) for text in texts]
        self.tfidf = gensim.models.TfidfModel(mm_corpus)

        # split into two corpii (haha) and ds?.setGensimCorpus(mm)
        ds1.setGensimCorpus(mm_corpus[0:ds1.size()])
        ds2.setGensimCorpus(mm_corpus[ds1.size():(ds1.size()+ds2.size())])

        self.lsi = gensim.models.LsiModel(self.tfidf[mm_corpus], id2word=dictionary, num_topics=params.LSI_TOPICS)
        # For some reason can't use this interchangeably:
        #   self.lsi = gensim.models.LdaModel(corpus=tfidf, num_topics=params.LSI_TOPICS)

    def getLSA(self):
        return self.lsi

    def getTfidf(self):
        return self.tfidf
