from FeatureBase import FeatureBase
import nltk
import numpy as np
import re
from spelling.SpellChecker import SpellChecker

class FeatureSpelling(object):

    spell_checker = SpellChecker()
    word_splitter = re.compile("\\s+")

    def __init__(self):
        self.features = np.array(())
        self.type = 'real'

    def numFeatures(self):
        """Return the number of features for this feature vector"""
        return length(self.features)

    def featureType(self):
        """Returns string description of the feature type, such as 'real-valued', 'binary', 'enum', etc."""
        return self.type

    def getFeatureMatrix(self):
        """Returns ordered list of features."""
        return self.features

    def extractFeatures(self, ds):
        """Extracts features from a DataSet ds"""
        lenfeats = list()
        for line in ds.getRawText():
            curfeat = list()
            words = (FeatureSpelling.word_splitter.split(line)) 
            word_set = set(words)
            misspelled_words = set()

            for word in word_set:
                suggestions = FeatureSpelling.spell_checker.extractSpellingSuggestions(word) 
                if suggestions is not None:
                    misspelled_words.add(word)

            curfeat.append(len(misspelled_words))
            curfeat.append(len(misspelled_words)/len(word_set)) 

        self.features = np.asarray(lenfeats)
        return

FeatureBase.register(FeatureSpelling)

