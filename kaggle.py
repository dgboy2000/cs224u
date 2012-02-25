import DataSet
from feature import FeatureHeuristics, Utils
from learn import LinearRegression
from score import KappaScore, MeanKappaScore

ds = DataSet.DataSet()
ds.importData('data/training_set_rel2.tsv')
ds.setTrainSet(True)

feat = FeatureHeuristics.FeatureHeuristics()
feat.extractFeatures(ds)

all_feats = list()
all_feats.append(feat)

mat = Utils.combine_features(ds, all_feats)
grades = ds.getGrades()

grades = mat[0,:] # TODO: set the grades
lr = LinearRegression(mat, grades)
lr.learn(intercept = True)
scores = [lr.predict(mat[i,:]) for i in mat.shape[0]]
predicted_grades = [lr.grade(score) for score in scores]

kappa = KappaScore(grades, predicted_grades).quadratic_weighted_kappa()
print "Yay! Achieved kappa score %f" %kappa


# 'mat' now contains all features in the matrix. Rows = instances, Columns = features.
