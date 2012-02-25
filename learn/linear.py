import numpy
from scipy import linalg


class LinearRegression:
    """Interface for scipy linear regression."""
    def __init__(self, X, Y):
        """def __init__(self, X, Y):
        
        X is a matrix where the i-th row corresponds to the i-th data point. Y is the output vector."""
        self.X = numpy.array(X)
        self.Y = numpy.array(Y)
        self.has_intercept = False

    def solve(self, intercept=False):
        """Solve the linear regression and save the parameters."""
        A = self.X
        self.has_intercept = intercept
        if self.has_intercept:
            A = numpy.hstack((A, numpy.ones((A.shape[0],1))))
        params, residues, rank, s = linalg.lstsq(A, self.Y)
        
        self.params = params
        if self.has_intercept:
            self.intercept = self.params[-1]
            self.params = self.params[:-1]

    def predict(self, x):
        """Predict y_hat given x"""
        assert len(self.params) > 0
        y_hat = numpy.dot(self.params, x)
        if self.has_intercept:
            y_hat += self.intercept
        return y_hat
        
    def grade(self, x, min_grade, max_grade):
        """Return an integer grade for x"""
        score = self.predict(x)
        grade = int(round(score))
        return max(min(max_grade, grade), min_grade)
        
    def set_curve(self, scores, grade_counts):
        """Find the score cutoffs that separate each of the different possible grades."""
        possible_grades = grade_counts.keys()
        possible_grades.sort()
        last_grade = possible_grades[0]
        for grade in possible_grades[1:]:
            if grade - last_grade != 1:
                raise "ERROR: did not specify count for grade %d; only saw these grades: %s" %(grade, str(possible_grades))
            last_grade = grade
        self.min_grade = possible_grades[0]
        self.max_grade = possible_grades[-1]
        
        if len(scores) != sum(grade_counts.values()):
            raise "ERROR: found %d scores and %d grades; must be same number" %(len(scores), sum(grade_counts.values()))
            
        num_scores = len(scores)
        scores.sort()
        num_lower_scores = 0
        self.cutoff_scores = []
        for grade in possible_grades[:-1]:
            num_lower_scores += grade_counts[grade]
            
            
            
            
            
            
            
            
            
            
            
        








