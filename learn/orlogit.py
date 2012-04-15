from LearnerBase import LearnerBase
from numpy import array

import rpy2
from rpy2.robjects.packages import importr
base = importr("base")
stats = importr('stats')
predict = stats.predict
from rpy2.robjects import DataFrame #*********FIX HERE***************

MASS = importr("MASS")
polr = MASS.polr

class OrLogit(object):
    """Interface for ordered logistic regression, implemented in R's MASS library"""

    def __init__(self, debug=False):
        self.debug = debug

    #def formatData(data_array):
        #data_array_t = data_array.transpose()
        

    def train(self, features, grades):
        formula = "grades ~ "
	dat_dict = {'grades': rpy2.robjects.FloatVector(grades)}
	
	# Each row of features is one essay. 
	# Transpose so a row is a feature
	feat_mat = features.transpose() 
	
	formula = "as.ordered(grades) ~ v0"
	dat_dict['v0'] = rpy2.robjects.FloatVector(feat_mat[0])
	
	# For each feature (column in original features)
	for i in range(1, feat_mat.shape[0]):
            dat_dict['v' + str(i)] = rpy2.robjects.FloatVector(feat_mat[i])
            formula += " + v" + str(i) 
        dat = DataFrame(dat_dict)

	if(self.debug):
            print "feat_mat dimensions", feat_mat.shape
            print "formula: ", formula
            #print "data: ", dat

	self.model = polr(formula, method="logistic", data=dat)
	#return polr(formula, method="logistic", data=dat)
	#if(self.debug):
            #print(self.model)

    def grade(self, raw_data, rnd):
        raw_data_t = raw_data.transpose()
        new_dict = {}
        for i in range(raw_data_t.shape[0]):
            new_dict['v' + str(i)] = rpy2.robjects.FloatVector(raw_data_t[i])
        new_data = DataFrame(new_dict)
        return array(predict(self.model, newdata=new_data, type="c"))
