from curve import Curve
from LearnerBase import LearnerBase
from math import log
import numpy as np
from scipy import linalg

class LinearRegression(object):
    """Interface for scipy linear regression."""
    
    def __init__(self, intercept=False, debug=False, **kwargs):
        self.debug = debug
        self.has_intercept = intercept
        self.used_features = None

    def train(self, features, grades):
        """Solve the linear regression and save the parameters. Set the curve to get the right
        grade distribution on the training data."""
        
        self.features = np.array(features)
        self.grades = np.array(grades)
        num_samples, num_features = self.features.shape
        
        if self.debug:
            print "Training linear model with %d features on %d essays" %(num_features, num_samples)
            
        correlations = self.get_feature_grade_correlations(self.features, self.grades)
        self.variable_order = sorted(enumerate(correlations), key=lambda tup: -tup[1])
        
        
        # # Do feature selection
        # best_ind = 0
        # features = np.transpose(np.array((self.features[:,self.variable_order[0][0]],)))
        # best_score = self.get_bic_score(features, self.grades)
        # for feat_ind in range(1, num_features):
        #     features = np.hstack((features, np.transpose(np.array((self.features[:,self.variable_order[feat_ind][0]],)))))
        #     score = self.get_bic_score(features, self.grades)
        #     if self.debug:
        #         print "Model with %d features achieved BIC score %f" %(feat_ind+1, score)
        #     if score < best_score:
        #         best_score = score
        #         best_ind = feat_ind
        # if self.debug:
        #     print "Best model with %d features achieved BIC score %f" %(best_ind+1, best_score)
        # self.used_features = sorted([tup[0] for tup in self.variable_order[:best_ind+1]])
        self.used_features = range(num_features)
        
        best_features = self.get_feature_subset(self.features, self.used_features)    
        if self.debug:
            actual_score = self.get_bic_score(best_features, self.grades)
            np.testing.assert_approx_equal(best_score, actual_score, err_msg="Expected score %f but found score %f" %(best_score, actual_score))
        
        if self.has_intercept:
            params, residues, rank, s = linalg.lstsq(np.hstack((best_features, np.ones((num_samples,1)))), self.grades)
        else:
            params, residues, rank, s = linalg.lstsq(best_features, self.grades)
        
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
        
    def get_best_n_features(self, features, n):
        """Return a sub-matrix of the top n columns/features, as specified by the variable_order."""
        best_features = np.transpose(np.array((features[:,self.variable_order[0][0]],)))
        for feat_ind in range(1, n):
            best_features = np.hstack((best_features, np.transpose(np.array((features[:,self.variable_order[feat_ind][0]],)))))
        return best_features
        
    def get_feature_subset(self, features, feat_inds):
        """Return a sub-matrix of the specified columns/features."""
        extracted_features = np.transpose(np.array((features[:,feat_inds[0]],)))
        for feat_ind in feat_inds[1:]:
            extracted_features = np.hstack((extracted_features, np.transpose(np.array((features[:,feat_ind],)))))
        return extracted_features
        
    def get_feature_grade_correlations(self, features, grades):
        """Returns a vector of the correlation of every feature with the grades, taken independently."""
        num_samples, num_features = features.shape
        covs = np.dot(grades - np.ones(num_samples)*np.mean(grades), features) / num_samples;
        denominator = np.std(features, axis=0) * np.std(grades)
        if (denominator == 0).any():
            print "WARNING: there is at least one constant feature"
            # import pdb;pdb.set_trace()
        return np.abs(covs / denominator)
        
    def get_bic_score(self, features, grades):
        """Compute and return the BIC score for the specified features on the specified grades.
        
        BIC = sum of squared errors + k*log(n)
        """
        num_essays = features.shape[0]
        if self.has_intercept:
            params, sq_error, rank, s = np.linalg.lstsq(np.hstack((features, np.ones((num_essays,1)))), grades)            
        else:
            params, sq_error, rank, s = np.linalg.lstsq(features, grades)

        # Compute the sq_error since sometimes numpy fails to return it
        sq_error = 0
        for essay_ind in range(num_essays):
            if self.has_intercept:
                sq_error = sq_error + (params[-1] + np.dot(params[:-1], features[essay_ind, :]) - grades[essay_ind]) ** 2
            else:
                sq_error = sq_error + (np.dot(params, features[essay_ind, :]) - grades[essay_ind]) ** 2
                
        return num_essays * log(sq_error / num_essays) + len(params) * log(num_essays)

    def grade(self, features, options={}):
        """Return an integer grade for each feature vector in the specified array"""
        if "round" in options and options["round"]:
            min_grade = min(self.grades)
            max_grade = max(self.grades)
            return [self.grade_by_rounding(features[i, :], min_grade, max_grade) for i in range(features.shape[0])]
        return [self.curve.curve(self.predict(features[i, :])) for i in range(features.shape[0])]
            
    def predict(self, x):
        """Predict y_hat given x"""
        assert len(self.params) > 0
        if self.used_features is not None:
            x = [x[i] for i in self.used_features]
        y_hat = np.dot(self.params, x)
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
 
            
            
            
            
            
            
        








