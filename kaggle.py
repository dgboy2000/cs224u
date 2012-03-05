import DataSet, Corpus
from feature import FeatureHeuristics, FeatureSpelling, Utils, FeatureBigram, FeatureUnigram, FeaturePOSUnigram, FeaturePOSBigram
from learn import LinearRegression, SVM
from score import KappaScore, MeanKappaScore
import math

def extract(ds, corpus):
    feat = FeatureHeuristics.FeatureHeuristics()
    feat.extractFeatures(ds)
    
    spelling_feat = FeatureSpelling.FeatureSpelling()
    spelling_feat.extractFeatures(ds)

    #bigram_feat = FeatureBigram.FeatureBigram()
    #bigram_feat.extractFeatures(ds, corpus)

    #bigram_pos_feat = FeaturePOSBigram.FeaturePOSBigram()
    #bigram_pos_feat.extractFeatures(ds, corpus)

    unigram_feat = FeatureUnigram.FeatureUnigram()
    unigram_feat.extractFeatures(ds, corpus)

    unigram_pos_feat = FeaturePOSUnigram.FeaturePOSUnigram()
    unigram_pos_feat.extractFeatures(ds, corpus)

    all_feats = list()
    all_feats.append(feat)
    all_feats.append(spelling_feat)
    #all_feats.append(bigram_feat)
    #all_feats.append(bigram_pos_feat)
    all_feats.append(unigram_feat)
    all_feats.append(unigram_pos_feat)

    mat = Utils.combine_features(ds, all_feats)
    return mat
    
def learn(ds, mat):
    grades = ds.getGrades()

    learner = LinearRegression(intercept = True)
    #learner = SVM()
    learner.train(mat, grades)
    
    return learner

def eval(mat, learner, ds):
    grades = ds.getGrades()
    round = False
    if ds.getEssaySet() == 8:
        round = True
    predicted_grades = learner.grade(mat, {'round': round})

    kappa = KappaScore(grades, predicted_grades)
    print "Kappa Score %f" %kappa.quadratic_weighted_kappa()

    if not ds.isTrainSet():
        i = 0
        lines = ds.getRawText()
        f = open('output/val.set%d.domain%d.tsv' % (ds.getEssaySet(), ds.getDomain()), 'w')
        for grade in grades:
            pgrade = predicted_grades[i]
            line = lines[i]
            f.write('%d\t%d\t%d\t%s\n' % (math.fabs(pgrade-grade), grade, pgrade, line))
            i+=1

    return kappa

train_mean_kappa = MeanKappaScore()
val_mean_kappa = MeanKappaScore()

for essay_set in range(1, 9):
    total_domains = 1
    if essay_set == 2:
        total_domains = 2

    for domain in range(1, total_domains+1):
        ds_train = DataSet.DataSet()
        ds_train.importData('data/c_train.tsv', essay_set=essay_set, domain_id=domain)
        ds_train.setTrainSet(True)
        ds_train.setID('c_rand')

        ds_val = DataSet.DataSet()
        ds_val.importData('data/c_val.tsv', essay_set=essay_set, domain_id=domain)
        ds_val.setTrainSet(False)
        ds_val.setID('c_rand')

        corpus = Corpus.Corpus()
        corpus.setCorpus('ds', ds_train)

        if (len(ds_train.getRawText()) > 0 and len(ds_val.getRawText())> 0):
            mat_train = extract(ds_train, corpus)
            mat_val = extract(ds_val, corpus)

            model = learn(ds_train, mat_train)

            print "Train / Test (Essay Set #%d, Domain #%d)" % (essay_set, domain)
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
    ds_train.setID('full_kaggle')

    ds_test = DataSet.DataSet()
    ds_test.importData('data/valid_set.tsv', essay_set, domain_id)
    ds_test.setTrainSet(False)
    ds_test.setID('full_kaggle')

    corpus = Corpus.Corpus()
    corpus.setCorpus('ds', ds_train)

    if (ds_train.size() > 0 and ds_test.size() > 0):
        mat_train = extract(ds_train, corpus)
        mat_test = extract(ds_test, corpus)

        model = learn(ds_train, mat_train)

        predicted_grades = model.grade(mat_test)

        ds_test.outputKaggle(predicted_grades, fd)

# Hi - please don't uncomment the function above. If you don't want to run this, just set the following flag to false.

RUN_KAGGLE = False

if RUN_KAGGLE:
    fd = open('data/kaggle_out.tsv', 'w')
    fd.write('prediction_id\tessay_id\tessay_set\tessay_weight\tpredicted_score\n')
    for essay_set in range(1, 9):
        run_test(essay_set, 1, fd)
    run_test(2, 2, fd) # essay set 2 is the only one with domain 2 scores

