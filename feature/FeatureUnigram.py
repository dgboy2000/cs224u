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

class FeatureUnigram(object):

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

        unigram_scored_fname = 'cache/unigram_corpus_scored.%s.set%d.pickle' % (ds.getID(), ds.getEssaySet())
        try:
            f = open(unigram_scored_fname, 'rb')
            print "Found Pickled <Unigram Corpus Scores>. Loading..."
            scored = pickle.load(f)
        except:
            freqs = nltk.FreqDist(x.lower() for x in corpus.getWords()) # should we preserve case?
            scored = freqs.items()
            scored = [(word, score) for (word, score) in scored if not word in nltk.corpus.stopwords.words('english')]
            pickle.dump(scored, open(unigram_scored_fname, 'wb'))

        scored = scored[0:params.TOTAL_WORD_UNIGRAMS]

        feats = list()
        for line in ds.getRawText():
            tokens = LanguageUtils.tokenize(line)

            cur_feats = list()
            for word, score in scored:
                if word in tokens:
                    cur_feats.append(1)
                else:
                    cur_feats.append(0)

            feats.append(cur_feats)

        self.features = np.asarray(feats)
        return

FeatureBase.register(FeatureUnigram)

