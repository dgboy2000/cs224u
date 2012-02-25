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

    def test_set_curve(self):
        lr = LinearRegression([],[])

        scores = numpy.array(range(4))
        grade_counts = {1:1,2:1,3:1,4:1}
        lr.set_curve(scores, grade_counts)
        self.assertEqual([0.5,1.5,2.5], lr.cutoff_scores)

        scores = numpy.array(range(6))
        grade_counts = {1:1,2:2,3:2,4:1}
        lr.set_curve(scores, grade_counts)
        self.assertEqual([0.5,2.5,4.5], lr.cutoff_scores)


if __name__ == '__main__':
    unittest.main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
