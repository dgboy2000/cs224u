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

class FeatureNN(object):

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

        ds_train = corpus.getTrain()
        tlen = len(ds_train.getRawText())
        tgrades = ds_train.getGrades()

        index = gensim.similarities.MatrixSimilarity(lsi[tfidf[ds_train.getGensimCorpus()]])
        for mm in ds.getGensimCorpus():
            cur_feat = list()
            vec_lsi = lsi[tfidf[mm]]

            sims = index[vec_lsi]

            # Loop in order of most similar.
            # Keep a counter of how many added.
            # Only add if index < length(ds_train)
            sorted_inds = [i[0] for i in sorted(enumerate(sims), key=lambda x:x[1], reverse=True)]

            counts = dict()
            for grade in range(min(tgrades), max(tgrades)+1):
                counts[grade] = 0
                
            for i in range(params.NUM_NN):
                if i == 0 and ds.isTrainSet():
                    continue
                ind = sorted_inds[i]
                grade = tgrades[ind]
                counts[grade] += 1

            if ds.isTrainSet():
                ind = sorted_inds[params.NUM_NN]
                grade = tgrades[ind]
                counts[grade] += 1

            for key in counts:
                cur_feat.append(counts[key])

            feats.append(cur_feat)

        self.features = np.asarray(feats)
        return

FeatureBase.register(FeatureNN)

