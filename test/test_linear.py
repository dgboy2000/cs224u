from learn import LinearRegression
import numpy
import unittest

class Test_linear_regression(unittest.TestCase):

    def test_solve(self):
        X = [[1, 0], [0, 2]]
        Y = [3, 2];
        lr = LinearRegression(X, Y)
        lr.solve()
        self.assertFalse(lr.has_intercept)
        self.assertAlmostEqual(3, lr.params[0])
        self.assertAlmostEqual(1, lr.params[1])
        self.assertEqual(2, len(lr.params))
        
        X = [[1, 0], [0, 2], [0, 3]]
        Y = [3, 2, 1];
        lr = LinearRegression(X, Y)
        lr.solve(intercept=True)
        self.assertTrue(lr.has_intercept)
        self.assertAlmostEqual(-1, lr.params[0])
        self.assertAlmostEqual(-1, lr.params[1])
        self.assertEqual(2, len(lr.params))
        self.assertAlmostEqual(4, lr.intercept)

    def test_predict(self):
        X = [[1, 0], [0, 2]]
        Y = [3, 2];
        lr = LinearRegression(X, Y)
        lr.solve()
        
        self.assertAlmostEqual(3, lr.predict([1,0]))
        self.assertAlmostEqual(4, lr.predict([1,1]))




if __name__ == '__main__':
    unittest.main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
