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

class FeaturePOSBigram(object):

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

        # NLTK setup
        bigram_measures = nltk.collocations.BigramAssocMeasures()

        bigram_scored_fname = 'cache/bigram_pos_corpus_scored.set%d.pickle' % ds.getEssaySet()
        try:
            f = open(bigram_scored_fname, 'rb')
            print "Found Pickled <Bigram POS Corpus Scores>. Loading..."
            scored = pickle.load(f)
        except:
            finder = BigramCollocationFinder.from_words(corpus.getPOS())
            scored = finder.score_ngrams(bigram_measures.raw_freq)
            pickle.dump(scored, open(bigram_scored_fname, 'wb'))

        scored = scored[0:params.TOTAL_POS_BIGRAMS]

        feats = list()
        for tokens in ds.getPOS():
            finder = BigramCollocationFinder.from_words(tokens)
            cur_scored = finder.score_ngrams(bigram_measures.raw_freq)

            bigram_scores = dict()
            for bigram, score in cur_scored:
                bigram_scores[bigram] = score

            cur_feats = list()
            for bigram, score in scored:
                if bigram in bigram_scores:
                    cur_feats.append(bigram_scores[bigram]) # could switch to binary 1/0
                else:
                    cur_feats.append(0.0)

            feats.append(cur_feats)

        self.features = np.asarray(feats)
        return

FeatureBase.register(FeaturePOSBigram)

