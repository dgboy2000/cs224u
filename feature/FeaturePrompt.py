import abc
from FeatureBase import FeatureBase
import nltk
from nltk.collocations import *
import Corpus
import os
import cPickle as pickle
import params
import numpy as np
import LanguageUtils
import gensim

class FeaturePrompt(object):

    def __init__(self):
        self.features = list()
        self.type = 'undefined'

    def numFeatures(self):
        """Return the number of features for this feature vector"""
        return length(self.features)

    def featureType(self):
        """Returns string description of the feature type, such as 'real-valued', 'binary', 'enum', etc."""
        return self.type

    def getFeatureMatrix(self):
        """Returns ordered list of features."""
        return self.features

    def extractFeatures(self, ds, corpus):
        """Extracts features from a DataSet ds"""

        # load into memory the string from data/essay_set_desc_?.txt
        f = open('data/essay_set_desc_%d.txt' % ds.getEssaySet(), 'r')
        prompt = f.read()

        # tokenize into unigrams & bigrams
        tokens = LanguageUtils.tokenize(prompt)

        bigram_measures = nltk.collocations.BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(tokens)
        scored = finder.score_ngrams(bigram_measures.raw_freq)
        for bigram, score in scored:
            tokens.append(bigram)

        # Get feature bows for projection into LSI
        dictionary = corpus.getWordDictionary()

        # get lsi
        lsi = corpus.getLSA()
        tfidf = corpus.getTfidf()
        mm_corpus = ds.getGensimCorpus()

        # project into lsi space
        vec_bow = dictionary.doc2bow(tokens)
        vec_lsi = lsi[tfidf[vec_bow]]

        index = gensim.similarities.MatrixSimilarity(lsi[tfidf[ds.getGensimCorpus()]])

        sims = index[vec_lsi]

        feats = list()
        for sim in sims:
            cur_feat = list()
            cur_feat.append(sim)
            feats.append(cur_feat)

        self.features = np.asarray(feats)
        return

FeatureBase.register(FeaturePrompt)

