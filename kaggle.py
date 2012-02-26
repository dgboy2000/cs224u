import DataSet
from feature import FeatureHeuristics, FeatureSpelling, Utils
from learn import LinearRegression
from score import KappaScore, MeanKappaScore

def extract(ds):
    feat = FeatureHeuristics.FeatureHeuristics()
    feat.extractFeatures(ds)
    
    spelling_feat = FeatureSpelling.FeatureSpelling()
    spelling_feat.extractFeatures(ds)

    all_feats = list()
    all_feats.append(feat)
    all_feats.append(spelling_feat)

    mat = Utils.combine_features(ds, all_feats)
    return mat
    
def learn(ds, mat):
    grades = ds.getGrades()

    lr = LinearRegression(mat, grades)
    lr.solve(intercept = True)
    
    scores = [lr.predict(mat[i,:]) for i in range(mat.shape[0])]
    grade_counts = {}
    for grade in grades:
        if grade not in grade_counts:
            grade_counts[grade] = 0
        grade_counts[grade] += 1
    
    lr.set_curve(scores, grade_counts)
    return lr

def eval(mat, lr, ds):
    grades = ds.getGrades()
    #predicted_grades = [lr.grade_by_rounding(mat[i,:], min(grades), max(grades)) for i in range(mat.shape[0])]
    predicted_grades = [lr.grade(mat[i,:]) for i in range(mat.shape[0])]

    kappa = KappaScore(grades, predicted_grades)
    print "Kappa Score %f" %kappa.quadratic_weighted_kappa()

    return kappa

train_mean_kappa = MeanKappaScore()
test_mean_kappa = MeanKappaScore()

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
        train_mean_kappa.add(eval(mat_train, model, ds_train))
        test_mean_kappa.add(eval(mat_val, model, ds_val))
        print "--\n"

print "Overall Train / Test"
print "Kappa Score %f" %train_mean_kappa.mean_quadratic_weighted_kappa()
print "Kappa Score %f" %test_mean_kappa.mean_quadratic_weighted_kappa()


