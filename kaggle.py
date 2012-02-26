import DataSet
from feature import FeatureHeuristics, Utils
from learn import LinearRegression
from score import KappaScore, MeanKappaScore

def extract(ds):
    feat = FeatureHeuristics.FeatureHeuristics()
    feat.extractFeatures(ds)

    all_feats = list()
    all_feats.append(feat)

    mat = Utils.combine_features(ds, all_feats)
    return mat
    
def learn(ds, mat):
    lr = LinearRegression(mat, ds.getGrades())
    lr.solve(intercept = True)
    return lr

def eval(mat, lr, ds):
    grades = ds.getGrades()
    predicted_grades = [lr.grade_by_rounding(mat[i,:], min(grades), max(grades)) for i in range(mat.shape[0])]

    kappa = KappaScore(grades, predicted_grades).quadratic_weighted_kappa()
    print "Kappa Score %f" %kappa

for essay_set in range(1, 9):
    ds_train = DataSet.DataSet()
    ds_train.importData('data/c_train.tsv', essay_set)
    ds_train.setTrainSet(True)

    ds_val = DataSet.DataSet()
    ds_val.importData('data/c_val.tsv', essay_set)
    ds_val.setTrainSet(False)

    if (len(ds_train.getRawText()) > 0 and len(ds_val.getRawText())> 0):
        mat_train = extract(ds_train)
        mat_val = extract(ds_val)

        model = learn(ds_train, mat_train)

        print "Train / Test"
        eval(mat_train, model, ds_train)
        eval(mat_val, model, ds_val)
        print "--\n"
