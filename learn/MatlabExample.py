from LearnerBase import LearnerBase
import numpy as np
import os

class MatlabExample(object):

    def train(self, features, grades, essay_set, domain):
        """Train the learner on the specified features and grades."""
        cmd = 'echo "prefix=\'output/features.set%d.dom%d\';prefix2=\'output/ds.set%d.dom%d\';addpath(genpath(\'matlab\'));trainTestEssayPipe" | matlab -nodesktop' % (essay_set, domain, essay_set, domain)
        print cmd
        os.system(cmd)

        return

    def grade(self, features, essay_set, domain, options={}):
        """Return an integer grade for each feature vector in the specified array."""

        f = open('output/ds.set%d.dom%d' % (essay_set, domain), 'r') # TODO this is broken
        grades = list()
        for line in f.readlines():
            grades.append(int(line))

        return np.asarray(grades)


LearnerBase.register(MatlabExample)




