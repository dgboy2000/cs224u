from FeatureBase import FeatureBase
import nltk
import numpy as np
import os
import pickle
from spelling.CreateDictionary import CreateDictionary
from spelling.WordCounter import WordCounter

class FeatureSpelling(object):

    word_splitter = nltk.WordPunctTokenizer()
    dictionary = CreateDictionary().getDictionary()
    word_count = WordCounter().getCounts()

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

    def extractFeatures(self, ds, corpus):
        """Extracts features from a DataSet ds"""
        lenfeats = list()
        for line in ds.getRawText():
            curfeat = list()
            words = FeatureSpelling.word_splitter.tokenize(line) 
            words = [word.lower() for word in words]
            words = [word for word in words if word.isalpha()]
            word_set = set(words)
            word_set.discard('')
            misspelled_words = set()
            misspelled_word_count = 0
            misspelled_char_count = 0
            misspelled_char_count_in_set = 0
            misspelled_short_word_count = 0

            for word in words:
                if not word in FeatureSpelling.dictionary:
                    misspelled_word_count += 1
                    if len(word) < 6:
                        misspelled_short_word_count += 1
                    misspelled_char_count += len(word)

            for word in word_set:
                if not word in FeatureSpelling.dictionary:
                    misspelled_words.add(word)
                    misspelled_char_count_in_set += len(word)
            
            curfeat.append(len(misspelled_words))
            curfeat.append(misspelled_short_word_count)
            #curfeat.append(len(misspelled_words)/len(word_set)) 
            #curfeat.append(misspelled_word_count)
            curfeat.append(misspelled_char_count / (misspelled_word_count+1))
            #curfeat.append(misspelled_word_count / len(words))
            #curfeat.append(misspelled_char_count_in_set / (len(word_set) + 1))
            
            lenfeats.append(curfeat)

        self.features = np.asarray(lenfeats)
        return

FeatureBase.register(FeatureSpelling)

