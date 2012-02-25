import abc
from FeatureBase import FeatureBase
import nltk
import numpy as np

class FeatureHeuristics(object):

    def __init__(self):
        self.features = np.array(())
        self.type = 'real'

    def numFeatures(self):
        """Return the number of features for this feature vector"""
        return length(self.features)

    def featureType(self):
        """Returns string description of the feature type, such as 'real-valued', 'binary', 'enum', etc."""
        return self.type

    def getInstanceFeatures(self, lineNum):
        """Returns numpy array of features."""
        return

    def getFeatureMatrix(self):
        """Returns numpy matrix of features"""
        return self.features

    def extractFeatures(self, ds):
        """Extracts features from a DataSet ds"""
        lenfeats = list()
        for line in ds.getRawText():
            curfeat = list()
            curfeat.append(len(line))
            lenfeats.append(curfeat)

        self.features = np.asarray(lenfeats)
        return

FeatureBase.register(FeatureHeuristics)

