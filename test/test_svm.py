from learn import SVM, write_dataset
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
            
    def setUp(self):
        os.system('cd learn && make > /dev/null')

    def test_simple(self):
        test_data_filename = '/tmp/cs224u_features_test.dat'
        test_model_filename = '/tmp/cs224u_model_test.dat'
        test_scores_filename = '/tmp/cs224u_scores_test'

        num_grades = 10
        grades, features = self.simpleTestGradesAndFeatures(num_grades)
        write_dataset(test_data_filename, features, grades)

        os.system('learn/rank_svm %s %s > /dev/null' %(test_data_filename, test_model_filename))
        os.system('learn/rank_svm %s %s %s > /dev/null' %(test_data_filename, test_model_filename, test_scores_filename))

        scores = [(ind, float(score)) for ind,score in enumerate(open(test_scores_filename).readlines())]
        scores.sort(key=lambda tup: tup[1])
        ranking = [tup[0] for tup in scores]
        self.assertEqual(range(10), ranking)
        
    def test_svm_py(self):
        """Same as test_simple, but use the SVM class to drive the test."""
        svm = SVM()
        
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
