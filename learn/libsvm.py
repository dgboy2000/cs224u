from curve import Curve
import inspect
import numpy as np
import os
import sys

learn_path = os.path.dirname(inspect.getfile(inspect.currentframe()))
sys.path.append(os.path.join(learn_path, 'libsvm-3.11/python'))
from svmutil import *


class LibSVM:
    """Python interface to regression svm in libsvm."""
    
    linear_params = '-c 10 -t 0 -s 3'
    quadratic_params = '-c 200 -s 3 -t 1 -r 1 -d 2'
    cubic_params = '-c 100 -s 3 -t 1 -r 0.4 -d 3'
    
    def __init__(self):
        self.model = None
    def train(self, features, grades):
        """Train a rank_svm on the specified features and grades.
        train_rank_svm(self, features, grades):
        
        features - numpy array/matrix with one row per essay
        grades - vector with one entry per essay
        """
        self.grades = grades
        self.features = features
        num_essays, num_features = features.shape

        training_data = self.format_features(features)
        self.model = svm_train(list(grades), training_data, LibSVM.linear_params)
        
        grade_counts = {}
        for grade in grades:
            if grade not in grade_counts:
                grade_counts[grade] = 0
            grade_counts[grade] += 1
        self.grade_probs = dict([(grade, count/float(num_essays)) for grade,count in grade_counts.iteritems()])
        scores = self.predict(features)
        self.curve = Curve(scores, probs=self.grade_probs)
        
    def grade(self, features, options={}):
        scores = self.predict(features)
        if "round" in options and options["round"]:
            min_grade = min(self.grades)
            max_grade = max(self.grades)
            return [self.grade_by_rounding(score, min_grade, max_grade) for score in scores]
        return [self.curve.curve(score) for score in scores]
        
    def grade_by_rounding(self, score, min_grade, max_grade):
        grade = int(round(score))
        return max(min(max_grade, grade), min_grade)
        
    def format_features(self, features):
        """Convert features into libsvm format [{1:feat1, 2:feat2, 3:feat3, ...}, {...},...]"""
        data = []
        for essay_ind in range(features.shape[0]):
            feature_dict = dict([(feat_ind+1,feat_val) for feat_ind,feat_val in enumerate(features[essay_ind,:])])
            data.append(feature_dict)
        return data
        
    def predict(self, features):
        """Run regression svm to score the specified essay features (numpy matrix/array).
        Returns a vector of scores of the specified essays."""
        assert self.model is not None
        return svm_predict(range(features.shape[0]), self.format_features(features), self.model)[0]
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
