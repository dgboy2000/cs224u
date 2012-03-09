from learn import RankSVM
import numpy
import os
import random
import unittest

class Test_svm(unittest.TestCase):
    """Test our rank svm implementation."""

    def simpleTestGradesAndFeatures(self, num_grades):
        grades = range(num_grades)
        features = []
        for grade in grades:
            r = num_grades * (1 - 2 * random.random())
            features.append([float(grade) / 2 + r, float(grade) / 2 - r])
        return grades, features
        
    def test_rank_svm(self):
        svm = RankSVM()
        
        num_grades = 15
        grades, features = self.simpleTestGradesAndFeatures(num_grades)
        features = numpy.asarray(features)
        svm.train(features, grades)
        scores = [(ind, float(score)) for ind,score in enumerate(svm.classify_rank_svm(features))]
        scores.sort(key=lambda tup: tup[1])
        ranking = [tup[0] for tup in scores]
        self.assertEqual(range(num_grades), ranking)
        


if __name__ == '__main__':
    unittest.main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
