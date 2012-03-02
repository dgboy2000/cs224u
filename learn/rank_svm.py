from curve import Curve
import numpy as np
import os
import sys

class SVM:
    """Python interface to rank SVM c code."""
    tmp_path = '/tmp'
    data_file = os.path.join(tmp_path, 'svm_rank_features.dat')
    model_file = os.path.join(tmp_path, 'svm_rank_model.dat')
    test_file = os.path.join(tmp_path, 'svm_rank_test.dat')
    predictions_file = os.path.join(tmp_path, 'svm_rank_predictions')
    
    def __init__(self):
        pass
    def train_rank_svm(self, features, grades):
        """Train a rank_svm on the specified features and grades.
        train_rank_svm(self, features, grades):
        
        features - numpy array/matrix with one row per essay
        grades - vector with one entry per essay
        """
        self.min_grade = min(grades)
        self.max_grade = max(grades)
        num_essays, num_features = features.shape
        
        f = open(SVM.data_file, 'w')
        f.write("%d\t%d\n" %(num_essays, num_features))
        for essay_ind in range(num_essays):
            feature_str = "\t".join([str(feat) for feat in features[essay_ind, :]])
            f.write("%d\t%s\n" %(grades[essay_ind], feature_str))
        f.close()
        
        os.system('learn/rank_svm %s %s > /dev/null' %(SVM.data_file, SVM.model_file))
        
        # Set a curve based on the SVM ranking scores
        grade_counts = {}
        for grade in grades:
            if grade not in grade_counts:
                grade_counts[grade] = 0
            grade_counts[grade] += 1
        self.grade_probs = dict([(grade, count/float(num_essays)) for grade,count in grade_counts.iteritems()])
        scores = self.classify_rank_svm(features)
        self.curve = Curve(scores, probs=self.grade_probs)
        
    def classify_rank_svm(self, features):
        """Run rank_svm to rank the specified essay features (numpy matrix/array).
        Returns a vector of scores of the specified essays."""
        num_essays, num_features = features.shape
        
        f = open(SVM.test_file, 'w')
        f.write("%d\t%d\n" %(num_essays, num_features))
        for essay_ind in range(num_essays):
            feature_str = "\t".join([str(feat) for feat in features[essay_ind, :]])
            f.write("0\t%s\n" %feature_str)
        f.close()
        
        os.system('learn/rank_svm %s %s %s > /dev/null' %(SVM.test_file, SVM.model_file, SVM.predictions_file))
        
        f = open(SVM.predictions_file)
        scores = [float(score) for score in f.readlines()]
        f.close()
        
        return scores
        
        # scores = [(ind, float(score)) for ind,score in enumerate(scores)]
        # scores.sort(key = lambda tup: tup[1]) # Sort in increasing rank order = decreasing score order
        # rankings = [0] * num_essays
        # cur_rank = 1
        # for ind,score in scores:
        #     rankings[ind] = cur_rank
        #     cur_rank += 1
        #     
        # return rankings
        
    def grade(self, feature):
        score = self.classify_rank_svm(np.reshape(feature, (1, len(feature))))
        return self.curve.curve(score)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            