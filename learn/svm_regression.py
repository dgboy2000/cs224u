from cStringIO import StringIO
from curve import Curve
import numpy as np
import os
import svmlight
import sys

class RegressionSVM:
    """Python interface to regression svm in svm_light."""
    
    def __init__(self, **kwargs):
        self.model = None
    def train(self, features, grades):
        """Train a rank_svm on the specified features and grades.
        train_rank_svm(self, features, grades):
        
        features - numpy array/matrix with one row per essay
        grades - vector with one entry per essay
        """
        self.min_grade = min(grades)
        self.max_grade = max(grades)
        num_essays, num_features = features.shape
        
        # Convert data into svmlight format [(label, [(feature, value), ...], query_id), ...]
        training_data = []
        for essay_ind,grade in enumerate(grades):
            feature_list = [(feat_ind+1,feat_val) for feat_ind,feat_val in enumerate(features[essay_ind,:])]
            training_data.append((grade, feature_list, 1))

        self.model = svmlight.learn(training_data, type='regression', kernel='polynomial', poly_degree=2, verbosity=0, C=100)
        
        grade_counts = {}
        for grade in grades:
            if grade not in grade_counts:
                grade_counts[grade] = 0
            grade_counts[grade] += 1
        self.grade_probs = dict([(grade, count/float(num_essays)) for grade,count in grade_counts.iteritems()])
        scores = self.classify_rank_svm(features)
        self.curve = Curve(scores, probs=self.grade_probs)
        
    def grade(self, features, options={}):
        scores = self.classify_rank_svm(features)
        return [self.curve.curve(score) for score in scores]
        
    def classify_rank_svm(self, features):
        """Run rank_svm to rank the specified essay features (numpy matrix/array).
        Returns a vector of scores of the specified essays."""
        assert self.model is not None

        # Convert data into svmlight format [(label, [(feature, value), ...], query_id), ...]
        test_data = []
        for essay_ind,feat_vec in enumerate(features):
            feature_list = [(feat_ind+1,feat_val) for feat_ind,feat_val in enumerate(feat_vec)]
            test_data.append((0, feature_list, 1))

        return svmlight.classify(self.model, test_data)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
