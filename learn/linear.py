from curve import Curve
from LearnerBase import LearnerBase
import numpy
from scipy import linalg


class LinearRegression(object):
    """Interface for scipy linear regression."""
    def __init__(self, intercept=False):
        self.has_intercept = intercept

    def train(self, features, grades):
        """Solve the linear regression and save the parameters. Set the curve to get the right
        grade distribution on the training data."""
        self.features = numpy.array(features)
        self.grades = numpy.array(grades)
        if self.has_intercept:
            params, residues, rank, s = linalg.lstsq(numpy.hstack((self.features, numpy.ones((self.features.shape[0],1)))), self.grades)
        else:
            params, residues, rank, s = linalg.lstsq(self.features, self.grades)
        
        self.params = params
        if self.has_intercept:
            self.intercept = self.params[-1]
            self.params = self.params[:-1]
            
        scores = [self.predict(self.features[i,:]) for i in range(self.features.shape[0])]
        grade_counts = {}
        for grade in grades:
            if grade not in grade_counts:
                grade_counts[grade] = 0
            grade_counts[grade] += 1

        self.set_curve(scores, grade_counts)

    def grade(self, feature_vec):
        """Return an integer grade for feature_vec"""
        return self.curve.curve(self.predict(feature_vec))
            
    def predict(self, x):
        """Predict y_hat given x"""
        assert len(self.params) > 0
        y_hat = numpy.dot(self.params, x)
        if self.has_intercept:
            y_hat += self.intercept
        return y_hat
        
    def grade_by_rounding(self, x, min_grade, max_grade):
        score = self.predict(x)
        grade = int(round(score))
        return max(min(max_grade, grade), min_grade)
            
    def set_curve(self, scores, grade_counts):
        """Set curve with histogram."""
        self.curve = Curve(scores, histogram=grade_counts)
            
LearnerBase.register(LinearRegression)
 
            
            
            
            
            
            
        








