from learn import LinearRegression
import numpy as np
import unittest

class Test_linear_regression(unittest.TestCase):

    def test_solve(self):
        X = [[1, 0], [0, 2]]
        Y = [3, 2]
        lr = LinearRegression()
        lr.train(X, Y)
        self.assertFalse(lr.has_intercept)
        self.assertAlmostEqual(3, lr.params[0])
        self.assertAlmostEqual(1, lr.params[1])
        self.assertEqual(2, len(lr.params))
        
        X = [[1, 0], [0, 2], [0, 3]]
        Y = [3, 2, 1]
        lr = LinearRegression(intercept=True)
        lr.train(X, Y)
        self.assertTrue(lr.has_intercept)
        self.assertAlmostEqual(-1, lr.params[0])
        self.assertAlmostEqual(-1, lr.params[1])
        self.assertEqual(2, len(lr.params))
        self.assertAlmostEqual(4, lr.intercept)

    def test_predict(self):
        X = [[1, 0], [0, 2]]
        Y = [3, 2]
        lr = LinearRegression()
        lr.train(X, Y)
        
        self.assertAlmostEqual(3, lr.predict([1,0]))
        self.assertAlmostEqual(4, lr.predict([1,1]))
        
    def test_correlations(self):
        X = np.array([[1, 1], [2, 3], [3, 1]])
        Y = np.array([1, 3, 1])
        lr = LinearRegression()
        correlations = lr.get_feature_grade_correlations(X, Y)
        expected = [0, 1]
        for feat_ind, corr in enumerate(correlations):
            self.assertAlmostEqual(expected[feat_ind], corr)


if __name__ == '__main__':
    unittest.main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
