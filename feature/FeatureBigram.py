import abc
from FeatureBase import FeatureBase
import nltk

### TODO do something with data/corpus.pickle

class FeatureBigram(object):

    def __init__(self):
        self.features = list()
        self.type = 'undefined'

    def numFeatures(self):
        """Return the number of features for this feature vector"""
        return length(self.features)

    def featureType(self):
        """Returns string description of the feature type, such as 'real-valued', 'binary', 'enum', etc."""
        return self.type

    def getFeatures(self, lineNum):
        """Returns ordered list of features."""
        return

    def extractFeatures(self, ds):
        """Extracts features from a DataSet ds"""
        return

FeatureBase.register(FeatureBigram)

