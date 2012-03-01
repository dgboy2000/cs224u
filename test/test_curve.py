from learn import Curve
import numpy
import unittest

class Test_curve(unittest.TestCase):

    def test_set_curve_with_histogram(self):
        scores = numpy.array(range(4))
        grade_counts = {1:1,2:1,3:1,4:1}
        curve = Curve(scores, histogram=grade_counts)
        self.assertEqual([0.5,1.5,2.5], curve.cutoff_scores)

        scores = numpy.array(range(6))
        grade_counts = {1:1,2:2,3:2,4:1}
        curve = Curve(scores, histogram=grade_counts)
        self.assertEqual([0.5,2.5,4.5], curve.cutoff_scores)
        
    def test_set_curve_with_probs(self):
        scores = numpy.array(range(4))
        grade_probs = {1:0.25,2:0.25,3:0.25,4:0.25}
        curve = Curve(scores, probs=grade_probs)
        self.assertEqual([0.5,1.5,2.5], curve.cutoff_scores)

        scores = numpy.array(range(6))
        grade_probs = {1:0.25,2:0,3:0.25,4:0.25,5:0.25}
        curve = Curve(scores, probs=grade_probs)
        self.assertEqual([0.5,0.5,2.5,3.5], curve.cutoff_scores)
        
    def test_curve(self):
        curve = Curve([1], histogram={1:1})
        curve.min_grade = 1
        curve.max_grade = 5
        curve.cutoff_scores = [0,1,1,2]
        
        self.assertEqual(1, curve.curve(-1))
        self.assertEqual(2, curve.curve(0.5))
        self.assertEqual(4, curve.curve(1.5))
        self.assertEqual(5, curve.curve(2.5))
        


if __name__ == '__main__':
    unittest.main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
