import DataSet
from feature import FeatureHeuristics, FeatureSpelling, Utils, FeatureBigram, FeatureUnigram
from learn import LinearRegression
from score import KappaScore, MeanKappaScore

def extract(ds):
    feat = FeatureHeuristics.FeatureHeuristics()
    feat.extractFeatures(ds)
    
    spelling_feat = FeatureSpelling.FeatureSpelling()
    spelling_feat.extractFeatures(ds)

    #bigram_feat = FeatureBigram.FeatureBigram()
    #bigram_feat.extractFeatures(ds)

    # unigram_feat = FeatureUnigram.FeatureUnigram()
    # unigram_feat.extractFeatures(ds)

    all_feats = list()
    all_feats.append(feat)
    all_feats.append(spelling_feat)
    #all_feats.append(bigram_feat)
    # all_feats.append(unigram_feat)

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
    # predicted_grades = [lr.grade_by_rounding(mat[i,:], min(grades), max(grades)) for i in range(mat.shape[0])]
    predicted_grades = [lr.grade(mat[i,:]) for i in range(mat.shape[0])]

    kappa = KappaScore(grades, predicted_grades)
    print "Kappa Score %f" %kappa.quadratic_weighted_kappa()

    return kappa

train_mean_kappa = MeanKappaScore()
val_mean_kappa = MeanKappaScore()

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
        val_mean_kappa.add(eval(mat_val, model, ds_val))
        print "--\n"

print "Overall Train / Test"
print "Kappa Score %f" %train_mean_kappa.mean_quadratic_weighted_kappa()
print "Kappa Score %f" %val_mean_kappa.mean_quadratic_weighted_kappa()

def run_test(essay_set, domain_id, fd):
    ds_train = DataSet.DataSet()
    ds_train.importData('data/training_set_rel3.tsv', essay_set, domain_id)
    ds_train.setTrainSet(True)

    ds_test = DataSet.DataSet()
    ds_test.importData('data/valid_set.tsv', essay_set, domain_id)
    ds_test.setTrainSet(False)

    if (ds_train.size() > 0 and ds_test.size() > 0):
        mat_train = extract(ds_train)
        mat_test = extract(ds_test)

        model = learn(ds_train, mat_train)

        predicted_grades = [model.grade(mat_test[i,:]) for i in range(mat_test.shape[0])]

        ds_test.outputKaggle(predicted_grades, fd)

fd = open('data/kaggle_out.tsv', 'w')
fd.write('prediction_id\tessay_id\tessay_set\tessay_weight\tpredicted_score\n')
for essay_set in range(1, 9):
    run_test(essay_set, 1, fd)
run_test(2, 2, fd) # essay set 2 is the only one with domain 2 scores

