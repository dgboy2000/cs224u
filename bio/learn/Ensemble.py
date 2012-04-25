import numpy as np
import params
import pickle
import Score

class Ensemble:
  def __init__(self, debug=False):
    self.debug = debug
    self.learners = []
    self.num_learners = 0
    self.learner_predictions = None
    self.weights = None
    
  def _crossValidateLearners(self, dataset, num_folds):
    for learner_ind,learner in enumerate(self.learners):
      learner_type = type(learner).__name__
      fname = 'cache/%s.pickle' %learner_type

      try:
        if (learner_type not in params.FEATURE_CACHE) or not params.FEATURE_CACHE[learner_type]:
          raise Exception('Do not use cache.')
        f = open(fname, 'rb')
        learner = pickle.load(f)
        if params.DEBUG:
          print "Using cached %s..." %learner_type
      except:
        if params.DEBUG:
          print "Cross-validating %s..." %learner_type
        learner.cross_validate(dataset, num_folds)
        pickle.dump(learner, open(fname, 'w'))

      self.learners[learner_ind] = learner
    
  def _getLearnerCVPredictions(self, dataset, num_folds):
    learner_predictions = np.zeros((dataset.getNumSamples(), len(self.learners)))
    dataset.createFolds(num_folds)
    for fold_ind in range(num_folds):
      if self.debug:
        print "Training learners on fold %d of %d..." %(fold_ind+1, num_folds)
      fold_train = dataset.getTrainFold(fold_ind)
      fold_test = dataset.getTestFold(fold_ind)
      prediction_inds = dataset.getTestFoldInds(fold_ind)
      self._trainLearners(fold_train.getFeatures(), fold_train.getLabels())
      for learner_ind,learner in enumerate(self.learners):
        learner_predictions[prediction_inds, learner_ind] = learner.predict(fold_test.getFeatures())
    return learner_predictions

  def _selectLearnerWeights(self, labels):
    """Grid search for the optimal weights on the params."""
    # TODO: this searches for weights over 2 models; search over more
    if self.debug:
      print "Optimizing ensemble weights..."
    assert self.num_learners == 2, "Can only weight 2 models at present"
    best_loss = float("inf")
    best_weight_a = None
    for weight_a in np.arange(0, 1, 0.01):
      combined_predictions = weight_a*self.learner_predictions[:, 0] + (1-weight_a)*self.learner_predictions[:, 1]
      cur_score = Score.Score(labels, combined_predictions)
      cur_loss = cur_score.getLogLoss()
      if cur_loss < best_loss:
        if self.debug:
          print "Achieved new best ensemble loss %f with weight_a %f" %(cur_loss, weight_a)
        best_loss = cur_loss
        best_weight_a = weight_a
    self.weights = np.asarray([best_weight_a, 1-best_weight_a])
    
  def _trainLearners(self, features, labels):
    """Train all learners on a speficied dataset"""
    for learner in self.learners:
      learner.train(features, labels)
        
  def addLearner(self, learner):
    self.learners.append(learner)
    self.num_learners += 1
    
  def train(self, dataset, num_folds):
    """Train the individual models and learn the ensemble weights."""
    if self.debug:
      print "Training ensemble..."
    self._crossValidateLearners(dataset, num_folds)
    self.learner_predictions = self._getLearnerCVPredictions(dataset, num_folds)
    self._selectLearnerWeights(dataset.getLabels())
    if self.debug:
      print "Training all models on all data..."
    self._trainLearners(dataset.getFeatures(), dataset.getLabels())
    
  def predict(self, dataset):
    probs = np.zeros(dataset.getNumSamples())
    for learner_ind,learner in enumerate(self.learners):
      probs += learner.predict(dataset.getFeatures()) * self.weights[learner_ind]
      
    return probs
  
  
  
  
  
  
  
  
  
  
  