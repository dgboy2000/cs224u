import DataSet, Corpus
from feature import FeatureHeuristics, FeatureSpelling, FeatureTransitions, Utils, FeatureBigram, FeatureUnigram, FeaturePOSUnigram, FeaturePOSBigram, FeatureLSI, FeaturePOS_LSI, FeaturePrompt
from learn import LinearRegression, SVM
from score import KappaScore, MeanKappaScore
import math

class Run:
    def __init__(self):
        self.train_feat_mat = None
        self.test_feat_mat = None
        self.train_pgrades = None
        self.test_pgrades = None
        self.model = None
        self.ds_train = None
        self.ds_test = None
        self.corpus = None

    def setup(self, train_fname, test_fname, essay_set, domain):
        self.ds_train = DataSet.DataSet()
        self.ds_train.importData(train_fname, essay_set=essay_set, domain_id=domain)
        self.ds_train.setTrainSet(True)

        self.ds_test = DataSet.DataSet()
        self.ds_test.importData(test_fname, essay_set=essay_set, domain_id=domain)
        self.ds_test.setTrainSet(False)

        self.corpus = Corpus.Corpus()
        self.corpus.setCorpus(self.ds_train, self.ds_test)

        return

    def _extract_ds(self, ds):
        feat = FeatureHeuristics.FeatureHeuristics()
        feat.extractFeatures(ds)
        
        spelling_feat = FeatureSpelling.FeatureSpelling()
        spelling_feat.extractFeatures(ds)

        transitions_feat = FeatureTransitions.FeatureTransitions()
        transitions_feat.extractFeatures(ds)

        lsi_feat = FeatureLSI.FeatureLSI()
        lsi_feat.extractFeatures(ds, self.corpus)

        pos_lsi_feat = FeaturePOS_LSI.FeaturePOS_LSI()
        pos_lsi_feat.extractFeatures(ds, self.corpus)

        #if ds.getEssaySet() == 3 or ds.getEssaySet() == 5:
        #    prompt_feat = FeaturePrompt.FeaturePrompt()
        #    prompt_feat.extractFeatures(ds, self.corpus)

        all_feats = list()
        all_feats.append(feat)
        all_feats.append(spelling_feat)
        all_feats.append(transitions_feat)
        all_feats.append(lsi_feat)
        all_feats.append(pos_lsi_feat)
        #if ds.getEssaySet() == 3 or ds.getEssaySet() == 5:
        #    all_feats.append(prompt_feat)

        return Utils.combine_features(ds, all_feats)

    def extract(self):
        self.corpus.genLSA()
        self.corpus.genPOS_LSA()

        self.train_feat_mat = self._extract_ds(self.ds_train)
        self.test_feat_mat = self._extract_ds(self.ds_test)
        return
        
    def learn(self):
        learner = LinearRegression(intercept = True)
        #learner = SVM()
        learner.train(self.train_feat_mat, self.ds_train.getGrades())
       
        self.model = learner
        return

    def predict(self):
        round = False
        if self.ds_train.getEssaySet() == 8:
            round = True
        self.train_pgrades = self.model.grade(self.train_feat_mat, {'round': round})
        self.test_pgrades = self.model.grade(self.test_feat_mat, {'round': round})
        return

    def _eval_ds(self, ds, pgrades):
        kappa = KappaScore(ds.getGrades(), pgrades)
        print "Kappa Score %f" %kappa.quadratic_weighted_kappa()

        """ TODO Later:
        if not ds.isTrainSet():
            i = 0
            lines = ds.getRawText()
            f = open('output/val.set%d.domain%d.tsv' % (ds.getEssaySet(), ds.getDomain()), 'w')
            for grade in grades:
                pgrade = predicted_grades[i]
                line = lines[i]
                f.write('%d\t%d\t%d\t%s\n' % (math.fabs(pgrade-grade), grade, pgrade, line))
                i+=1
        """

        return kappa

    def eval(self):
        print "Train/Test Scores: (ESSAY_SET #%d, DOMAIN %d)" % (self.ds_train.getEssaySet(), self.ds_train.getDomain())
        train_score = self._eval_ds(self.ds_train, self.train_pgrades)
        test_score = self._eval_ds(self.ds_test, self.test_pgrades)
        return train_score, test_score

    def outputKaggle(self, fd):
        self.ds_test.outputKaggle(self.test_pgrades, fd)
        return
