from learn import SVM, write_dataset
import numpy
import os
import random
import unittest

class Test_svm(unittest.TestCase):
    """Test our rank svm implementation."""

    def setUp():
        os.system('cd learn && make > /dev/null')

    def test_simple(self):
        num_grades = 10
        grades = range(num_grades)
        test_data_filename = '/tmp/cs224u_features_test.dat'
        test_model_filename = '/tmp/cs224u_model_test.dat'
        test_scores_filename = '/tmp/cs224u_scores_test'
        features = []
        for grade in grades:
            r = num_grades * (1 - 2 * random.random())
            features.append([float(grade) / 2 + r, float(grade) / 2 - r])
        write_dataset(test_data_filename, features, grades)

        os.system('learn/rank_svm %s %s > /dev/null' %(test_data_filename, test_model_filename))
        os.system('learn/rank_svm %s %s %s > /dev/null' %(test_data_filename, test_model_filename, test_scores_filename))

        scores = [(ind, float(score)) for ind,score in enumerate(open(test_scores_filename).readlines())]
        scores.sort(key=lambda tup: tup[1])
        ranking = [tup[0] for tup in scores]
        self.assertEqual(range(10), ranking)
        


if __name__ == '__main__':
    unittest.main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
