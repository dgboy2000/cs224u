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
        
    def _select_features_inclusive(self):
        """Greedily add features to select the optimal subset."""
        num_samples, num_features = self.features.shape
        
        features = np.ones((num_samples, 0))
        best_ind = 0
        best_score = float("inf")
        remaining_features = set(range(num_features))
        self.used_features = used_features = []
        while len(remaining_features) > 0 and len(self.used_features) + (1 if self.has_intercept else 0) < num_samples:
            best_iter_score = float("inf")
            for feat_ind in remaining_features:
                iter_features = np.hstack((features, np.transpose(np.array((self.features[:,feat_ind],)))))
                score = self.get_bic_score(iter_features, self.grades)
                # if self.debug:
                    # print "Iter %d: Model with feature %d achieved BIC score %f" %(len(used_features), feat_ind, score)
                if score < best_iter_score:
                    best_iter_score = score
                    best_iter_ind = feat_ind
            features = np.hstack((features, np.transpose(np.array((self.features[:,best_iter_ind],)))))
            remaining_features.remove(best_iter_ind)
            used_features.append(best_iter_ind)
            if self.debug:
                print "Best model with %d features achieves BIC score %f: %s" %(len(used_features), best_iter_score, str(used_features))
            if best_iter_score < best_score:
                best_score = best_iter_score
                self.used_features = list(used_features)
        if self.debug:
            print "Best model has %d features and achieves BIC score %f" %(len(self.used_features), best_score)
        
    def _select_features_exclusive(self):
        """Greedily remove features to select the optimal subset."""
        num_samples, num_features = self.features.shape

        features = self.features.copy()
        best_ind = 0
        best_score = self.get_bic_score(features, self.grades)
        used_features = range(num_features)
        while len(used_features) > 0:
            best_iter_score = float("inf")
            for feat_ind in range(len(used_features)):
                iter_features = np.hstack((features[:,:feat_ind], features[:,feat_ind+1:]))
                score = self.get_bic_score(iter_features, self.grades)
                # if self.debug:
                    # print "Iter %d: Model without feature %d achieved BIC score %f" %(num_features-len(used_features)+1, used_features[feat_ind], score)
                if score < best_iter_score:
                    best_iter_score = score
                    best_iter_ind = feat_ind
            features = np.hstack((features[:,:best_iter_ind], features[:,best_iter_ind+1:]))
            used_features = used_features[:best_iter_ind] + used_features[best_iter_ind+1:]
            if self.debug:
                print "Best model with %d features achieves BIC score %f: %s" %(len(used_features), best_iter_score, str(used_features))
            if best_iter_score < best_score:
                best_score = best_iter_score
                self.used_features = list(used_features)
        if self.debug:
            print "Best model has %d features and achieves BIC score %f" %(len(self.used_features), best_score)

    def select_features(self, options):
        """Do feature selection and save the list of optimal features."""
        # correlations = self.get_feature_grade_correlations(self.features, self.grades)
        # self.variable_order = [tup[0] for tup in sorted(enumerate(correlations), key=lambda tup: -tup[1])]
        
        if options['feature_selection'] == 'inclusive':
            self._select_features_inclusive()
        elif options['feature_selection'] == 'exclusive':
            self._select_features_exclusive()
        else:
            print "WARNING: no feature selection, using all features"
            self.used_features = range(num_features)

    def train(self, features, grades, options={}):
        """Solve the linear regression and save the parameters. Set the curve to get the right
        grade distribution on the training data."""
        
        self.features = np.array(features)
        self.grades = np.array(grades)
        self.min_grade = min(grades)
        self.max_grade = max(grades)
        num_samples, num_features = self.features.shape
        
        if self.debug:
            print "Training linear model with %d features on %d essays" %(num_features, num_samples)
            
        self.select_features(options)
        best_features = self.get_feature_subset(self.features, self.used_features)    
        
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
        
    # def get_best_n_features(self, features, n):
    #     """Return a sub-matrix of the top n columns/features, as specified by the variable_order."""
    #     best_features = np.transpose(np.array((features[:,self.variable_order[0]],)))
    #     for feat_ind in range(1, n):
    #         best_features = np.hstack((best_features, np.transpose(np.array((features[:,self.variable_order[feat_ind]],)))))
    #     return best_features
        
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
        return np.abs(covs / (denominator+1e-8))
        
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

    def _grade(self, feature_vec, options):
        """Return an integer grade for the specified feature vector."""
        if "round" in options and options["round"]:
            return self.grade_by_rounding(feature_vec, self.min_grade, self.max_grade)
        return self.curve.curve(self.predict(feature_vec))
        
    def grade(self, features, options={}):
        """Return integer grades for each feature vector in the specified array."""
        if len(features.shape) == 2:
            return [self._grade(features[i, :], options) for i in range(features.shape[0])]
        elif len(features.shape) == 1:
            return self._grade(features, options)
        raise TypeError("Features had bad shape: %s" %tr(features.shape))
            
    def predict(self, x):
        """Predict y_hat given x"""
        assert len(self.params) > 0
        if self.used_features is not None:
            x = [x[i] for i in self.used_features]
        y_hat = np.vdot(self.params, x)
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
 
            
            
            
            
            
            
        








