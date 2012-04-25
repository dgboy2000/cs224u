from LearnerBase import LearnerBase
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy import linalg
import Score

class RandomForest(object):
  def __init__(self, n_estimators=100, min_split=2, debug=False):
    self.rf = None
    self.debug = debug
    self.min_split = min_split
    self.n_estimators = n_estimators
    self.params = None
    
  def cross_validate(self, dataset, num_folds):
    pass
        
  def train(self, features, labels):
    self.rf = RandomForestClassifier(n_estimators=self.n_estimators, min_split=self.min_split)
    self.rf.fit(features, labels)
    
  def predict(self, features):
    num_samples, num_features = features.shape
    probs = [prob[1] for prob in self.rf.predict_proba(features)]
    return np.minimum(np.maximum(probs, 0.01*np.ones(num_samples)), 0.99*np.ones(num_samples))

    
LearnerBase.register(RandomForest)
    
    
    
    
    
    
    
    
    
    
    
