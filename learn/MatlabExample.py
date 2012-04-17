from LearnerBase import LearnerBase
import numpy as np
import os

class MatlabExample(object):

    def train(self, features, grades, essay_set, domain, option={}):
        """Train the learner on the specified features and grades."""
        cmd = 'echo "prefix=\'output/features.set%d.dom%d\';prefix2=\'output/ds.set%d.dom%d\';addpath(genpath(\'matlab\'));trainTestEssayPipe" | matlab -nodesktop -nosplash' % (essay_set, domain, essay_set, domain)
        print cmd
        os.system(cmd)

        return

    def predict(self, x):
        """ TODO fix this somehow. """
        return 0.0

    def grade(self, features, essay_set, domain, options={}):
        """Return an integer grade for each feature vector in the specified array."""

        f = open('output/ds.set%d.dom%d.%s.matOut' % (essay_set, domain, options['postfix']), 'r') # TODO this is broken
        grades = list()
        for line in f.readlines():
            grades.append(int(round(float(line))))

        return np.asarray(grades)


LearnerBase.register(MatlabExample)




