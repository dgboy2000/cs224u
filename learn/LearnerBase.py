import abc

class LearnerBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def train(self, features, grades, essay_set, domain):
        """Train the learner on the specified features and grades."""
        return

    @abc.abstractmethod
    def grade(self, features, essay_set, domain, options={}):
        """Return an integer grade for each feature vector in the specified array."""
        return


















