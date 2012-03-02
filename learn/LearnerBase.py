import abc

class LearnerBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def train(self, features, grades):
        """Train the learner on the specified features and grades."""
        return

    @abc.abstractmethod
    def grade(self, feature_vec):
        """Return an integer grade for the specified feature vector."""
        return


















