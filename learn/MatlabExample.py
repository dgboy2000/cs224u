from LearnerBase import LearnerBase
import numpy as np
import os
from curve import Curve

class MatlabExample(object):
    def __init__(self, **kwargs):
        """Ignore keyword arguments."""
        self.curve = None

    def train(self, features, grades, essay_set, domain, option={}):
        """Train the learner on the specified features and grades."""
        cmd = 'echo "prefix=\'output/features.set%d.dom%d\';prefix2=\'output/ds.set%d.dom%d\';addpath(genpath(\'matlab\'));trainTestEssayPipe" | matlab -nodesktop -nosplash -nojvm' % (essay_set, domain, essay_set, domain)
        print cmd
        os.system(cmd)

        # Get training values
        f = open('output/ds.set%d.dom%d.train.matOut' % (essay_set, domain), 'r')
        scores = list()
        for line in f.readlines():
            scores.append(float(line))

        grade_counts = {}
        for grade in grades:
            if grade not in grade_counts:
                grade_counts[grade] = 0
            grade_counts[grade] += 1

        self.set_curve(scores, grade_counts)

        return

    def predict(self, x):
        """ TODO fix this somehow. """
        return 0.0

    def grade(self, features, essay_set, domain, options={}):
        """Return an integer grade for each feature vector in the specified array."""

        f = open('output/ds.set%d.dom%d.%s.matOut' % (essay_set, domain, options['postfix']), 'r')
        scores = list()
        for line in f.readlines():
            scores.append(float(line))

        if options['round']:
            grades = [int(round(score)) for score in scores]
        else:
            grades = [self.curve.curve(score) for score in scores]

        return np.asarray(grades)

    def set_curve(self, scores, grade_counts):
        """Set curve with histogram."""
        self.curve = Curve(scores, histogram=grade_counts)


LearnerBase.register(MatlabExample)




