import numpy
from scipy import linalg


class LinearRegression:
    """Interface for scipy linear regression."""
    def __init__(self, X, Y):
        """def __init__(self, X, Y):
        
        X is a matrix where the i-th row corresponds to the i-th data point. Y is the intercept vector."""
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
        








