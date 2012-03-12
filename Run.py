import DataSet, Corpus
from feature import FeatureHeuristics, FeatureSpelling, FeatureTransitions, Utils, FeatureBigram, FeatureUnigram, FeaturePOSUnigram, FeaturePOSBigram, FeatureLSI, FeaturePOS_LSI, FeaturePrompt, FeatureSim
from learn import LinearRegression, SVM
import math
import os
import cPickle as pickle
from score import KappaScore, MeanKappaScore
import numpy as np
import params

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
        self.dist = None

    def setup(self, train_fname, test_fname, essay_set, domain, dist=None):
        self.ds_train = DataSet.DataSet(True)
        self.ds_train.importData(train_fname, essay_set=essay_set, domain_id=domain)

        self.ds_test = DataSet.DataSet(False)
        self.ds_test.importData(test_fname, essay_set=essay_set, domain_id=domain)

        self.corpus = Corpus.Corpus()
        self.corpus.setCorpus(self.ds_train, self.ds_test)

        self.dist = dist
        
        if params.DEBUG:
            print "BEGIN train/test: ESSAY_SET #%d, DOMAIN %d" % (self.ds_train.getEssaySet(), self.ds_train.getDomain())

        return

    def _extract_feat(self, ds, feat):
        feat_type = type(feat).__name__
        # do caching and stuff
        fname = 'cache/feature_mat.%s.%s.set%d.dom%d.pickle' % (
                 feat_type,
                 ds.getFilename(),
                 ds.getEssaySet(),
                 ds.getDomain())

        try:
            if (feat_type not in params.FEATURE_CACHE) or not params.FEATURE_CACHE[feat_type]:
                raise Exception('Do not use cache.')
            f = open(fname, 'rb')
            feat_mat = pickle.load(f)
        except:
            if params.DEBUG:
                print "Calculating ", feat_type
            feat.extractFeatures(ds, self.corpus)
            feat_mat = feat.getFeatureMatrix()
            pickle.dump(feat_mat, open(fname, 'w'))

        return feat_mat
        
    def _extract_ds(self, ds):
        all_feats = list()
        all_feats.append(self._extract_feat(ds, FeatureHeuristics.FeatureHeuristics()))
        all_feats.append(self._extract_feat(ds, FeatureSpelling.FeatureSpelling()))
        all_feats.append(self._extract_feat(ds, FeatureTransitions.FeatureTransitions()))
        all_feats.append(self._extract_feat(ds, FeatureLSI.FeatureLSI()))
        #all_feats.append(self._extract_feat(ds, FeatureSim.FeatureSim()))
        all_feats.append(self._extract_feat(ds, FeaturePOS_LSI.FeaturePOS_LSI()))
        #if ds.getEssaySet() == 3 or ds.getEssaySet() == 5:
        #   all_feats.append(self._extract_feat(ds, FeaturePrompt.FeaturePrompt()))

        return Utils.combine_features(ds, all_feats)

    def extract(self):
        #fname = 'cache/all_features.%s.%s.set%d.dom%d.pickle' % (
        #         self.ds_train.getFilename(),
        #         self.ds_test.getFilename(),
        #         self.ds_train.getEssaySet(),
        #         self.ds_train.getDomain())

        #try:
        #    f = open(fname, 'rb')
        #    print "Using pickled features for essay set %d, domain %d." % (
        #           self.ds_train.getEssaySet(), self.ds_train.getDomain())
        #    self.train_feat_mat, self.test_feat_mat = pickle.load(f)
        #except:
        self.corpus.genLSA()
        self.corpus.genPOS_LSA()

        self.train_feat_mat = self._extract_ds(self.ds_train)
        self.test_feat_mat = self._extract_ds(self.ds_test)

        #    f = open(fname, 'w')
        #    pickle.dump((self.train_feat_mat, self.test_feat_mat), f)

        """
        ## XXX: SPECIAL KIND OF FEATURE. It is dependent on the learner, so we will recalc these features
        ##      even if there is cache.
        other_pgrades = list()
        for other_grades in self.ds_train.getOtherDistsGrades():
            # TODO use raw predict score instead of the resolved one ??
            other_model = self._learn(self.train_feat_mat, other_grades)
            other_train_pgrades = np.asarray(
                [other_model.predict(self.train_feat_mat[i, :]) for i in range(self.train_feat_mat.shape[0])])
            other_test_pgrades = np.asarray(
                [other_model.predict(self.test_feat_mat[i, :]) for i in range(self.test_feat_mat.shape[0])])

            other_train_pgrades = np.asarray(other_train_pgrades).reshape(len(other_train_pgrades), 1)
            other_test_pgrades = np.asarray(other_test_pgrades).reshape(len(other_test_pgrades), 1)
            other_pgrades.append((other_train_pgrades,other_test_pgrades))

        # If final matrix is actually JUST the predicted grades of the other dists, it does better on *some* sets.
        self.train_feat_mat = np.zeros((self.train_feat_mat.shape[0], 0))
        self.test_feat_mat = np.zeros((self.test_feat_mat.shape[0], 0))
        for other_train_pgrades, other_test_pgrades in other_pgrades:
            self.train_feat_mat = np.concatenate((self.train_feat_mat, other_train_pgrades), axis=1)
            self.test_feat_mat = np.concatenate((self.test_feat_mat, other_test_pgrades), axis=1)
        """

        # Normalize to unit mean/var
        self.train_feat_mat = np.asarray(self.train_feat_mat, dtype=np.float) # convert all to float
        self.test_feat_mat = np.asarray(self.test_feat_mat, dtype=np.float)
        all_mat = np.concatenate((self.train_feat_mat, self.test_feat_mat), axis=0)
        for i in range(self.train_feat_mat.shape[1]): # norm to unit mean/var.
            if sum(all_mat[:,i] > 0.0):
                self.train_feat_mat[:,i] = (self.train_feat_mat[:,i] - np.mean(all_mat[:,i])) / np.var(all_mat[:,i])
        for i in range(self.test_feat_mat.shape[1]): # norm to unit mean/var.
            if sum(all_mat[:,i] > 0.0):
                self.test_feat_mat[:,i] = (self.test_feat_mat[:,i] - np.mean(all_mat[:,i])) / np.var(all_mat[:,i])

        return
        
    def _learn(self, feat_mat, grades):
        learner = LinearRegression(intercept = True)
        #learner = SVM()
        
        learner.train(feat_mat, grades)
        return learner

    def learn(self):
        self.model = self._learn(self.train_feat_mat, self.ds_train.getGrades())
        return

    def _predict(self, feat_mat, model):
        round = False
        if self.ds_train.getEssaySet() == 8:
            round = True

        return model.grade(feat_mat, {'round': round})

    def predict(self):
        self.train_pgrades = self._predict(self.train_feat_mat, self.model)
        self.test_pgrades = self._predict(self.test_feat_mat, self.model)
        return

    def _eval_ds(self, ds, pgrades):
        try:
            kappa = KappaScore(ds.getGrades(), pgrades)
        except:
            import pdb;pdb.set_trace()
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
