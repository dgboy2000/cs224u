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
import operator

class FeatureSim(object):

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

        feats = list()
        lsi = corpus.getLSA()
        tfidf = corpus.getTfidf()

        index = gensim.similarities.MatrixSimilarity(lsi[tfidf[ds.getGensimCorpus()]])
        for mm in ds.getGensimCorpus():
            cur_feat = list()
            vec_lsi = lsi[tfidf[mm]]

            sims = index[vec_lsi]

            # AGGREGATE SIMILARITY
            sims = index[vec_lsi]
            sims = np.asarray(sims)
            #cur_feat.append(np.mean(sims))
            #cur_feat.append(np.var(sims))
            cur_feat.append(np.mean(sims))
            #cur_feat.append(np.median(sims))

            feats.append(cur_feat)

        self.features = np.asarray(feats)
        return

FeatureBase.register(FeatureSim)

