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

class FeatureLSI(object):

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
        text = ds.getRawText()
        i=0
        for mm in ds.getGensimCorpus():
            cur_feat = list()
            if len(mm) == 0:
                if params.DEBUG:
                    print "WARNING: No LSI features for current feature. Using (0,1) - which may be a horrible assumption."
                mm = [(0,1)]

            for topic, score in lsi[tfidf[mm]]:
                cur_feat.append(score)

            if len(cur_feat) != params.LSI_TOPICS:
                print "NON-MATCHING FEATURE LENGTH...LSI"
                import pdb;pdb.set_trace()
            feats.append(cur_feat)
            i+=1

        self.features = np.asarray(feats)
        return

FeatureBase.register(FeatureLSI)

