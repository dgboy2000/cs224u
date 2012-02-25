import DataSet
from feature import FeatureHeuristics, Utils

ds = DataSet.DataSet()
ds.importData('data/training_set_rel2.tsv')
ds.setTrainSet(True)

feat = FeatureHeuristics.FeatureHeuristics()
feat.extractFeatures(ds)

all_feats = list()
all_feats.append(feat)

mat = Utils.combine_features(ds, all_feats)
grades = ds.getGrades()

# 'mat' now contains all features in the matrix. Rows = instances, Columns = features.
