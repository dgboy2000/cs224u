import abc
import DataSet

class FeatureBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def numFeatures(self):
        """Return the number of features for this feature vector"""
        return

    @abc.abstractmethod
    def featureType(self):
        """Returns string description of the feature type, such as 'real-valued', 'binary', 'enum', etc."""
        return

    @abc.abstractmethod
    def getFeatureMatrix(self):
        """Return numpy matrix of features."""
        return

    @abc.abstractmethod
    def extractFeatures(self, ds):
        """Extracts features from a DataSet ds"""
        return
