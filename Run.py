import DataSet, Corpus
from feature import FeatureHeuristics, FeatureSpelling, FeatureTransitions, Utils, FeaturePOSUnigram, FeaturePOSBigram, FeatureLSI, FeaturePOS_LSI, FeaturePrompt, FeatureSim
from learn import LinearRegression #, OrLogit, SVM
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
        learner = LinearRegression(intercept = True, debug = params.DEBUG)
        # learner = SVM(debug = params.DEBUG)
        # learner = OrLogit(debug = params.DEBUG)
        
        learner.train(feat_mat, grades, {'feature_selection': params.FEATURE_SELECTION})
        return learner
        
    def _learn_granular(self, feat_mat, grades):
        min_grade = min(grades)
        max_grade = max(grades)

        grade_to_model = {}
        cur_center_grade = min_grade + 1
        while cur_center_grade < max_grade:            
            inds_within_one_grade = [ind for ind,grade in enumerate(grades) if abs(grade-cur_center_grade) < 1.1]
            cur_feat_mat = np.vstack([feat_mat[ind, :] for ind in inds_within_one_grade])
            cur_grades = [grades[ind] for ind in inds_within_one_grade]
            
            learner = LinearRegression(intercept = True, debug = params.DEBUG)
            learner.train(cur_feat_mat, cur_grades, {'feature_selection': 'inclusive'})
            grade_to_model[cur_center_grade] = learner
            
            cur_center_grade += 1
        
        grade_to_model[min_grade] = grade_to_model[min_grade+1]
        grade_to_model[max_grade] = grade_to_model[max_grade-1]
        
        return grade_to_model

    def learn(self):
        self.model = self._learn(self.train_feat_mat, self.ds_train.getGrades())
        if params.GRANULAR_MODELS and self.ds_train.getEssaySet() != 8:
            self.granular_models = self._learn_granular(self.train_feat_mat, self.ds_train.getGrades())
        return

    def _predict(self, feat_mat, model):
        round = False
        if self.ds_train.getEssaySet() == 8:
            round = True

        return model.grade(feat_mat, {'round': round})
    
    def _predict_refine(self, raw_grades, feat_mat, model):
        refined_grades = [self.granular_models[grade].grade(feat_mat[ind, :], {'round': False}) for ind, grade in enumerate(raw_grades)]
        return refined_grades

    def predict(self):
        self.train_pgrades = self._predict(self.train_feat_mat, self.model)
        self.test_pgrades = self._predict(self.test_feat_mat, self.model)
        
        if params.GRANULAR_MODELS and self.ds_train.getEssaySet() != 8:
            self.train_pgrades = self._predict_refine(self.train_pgrades, self.train_feat_mat, self.model)
            self.test_pgrades = self._predict_refine(self.test_pgrades, self.test_feat_mat, self.model)
        return

    def _eval_ds(self, ds, pgrades):
        kappa = KappaScore(ds.getGrades(), pgrades)
        print "Kappa Score %f" %kappa.quadratic_weighted_kappa()


        ds_str = ''
        if ds.isTrainSet():
            ds_str = 'train'
            feat_mat = self.train_feat_mat
        else:
            ds_str = 'test'
            feat_mat = self.test_feat_mat

        real_pgrades = [self.model.predict(x) for x in feat_mat]


        i = 0
        lines = ds.getRawText()
        f = open('output/diffs.set%d.domain%d.%s' % (ds.getEssaySet(), ds.getDomain(), ds_str), 'w')
        f.write('#real_diff\tresolved_diff\tgt_grade\tpred_score\tpred_grade\tessay\n')
        for grade in ds.getGrades():
            pgrade = pgrades[i]
            try:
                real_pgrade = real_pgrades[i]
            except:
                import pdb; pdb.set_trace()
            line = lines[i]
            f.write('%f\t%d\t%d\t%f\t%d\t%s\n' % (math.fabs(real_pgrade-float(grade)), math.fabs(pgrade-grade), grade, real_pgrade, pgrade, line))
            i+=1

        return kappa

    def eval(self):
        print "Train/Test Scores: (ESSAY_SET #%d, DOMAIN %d)" % (self.ds_train.getEssaySet(), self.ds_train.getDomain())
        train_score = self._eval_ds(self.ds_train, self.train_pgrades)
        test_score = self._eval_ds(self.ds_test, self.test_pgrades)
        return train_score, test_score

    def outputKaggle(self, fd):
        self.ds_test.outputKaggle(self.test_pgrades, fd)
        return
        
    def _dump_feat_mat(self, filename, features):
        """Dump the specified feature matrix to human-readable file."""
        f = open(filename, 'w')
        num_essays, num_features = features.shape
        for essay_ind in range(num_essays):
            for feat_ind in range(num_features):
                f.write("%f\t" %features[essay_ind, feat_ind])
            f.write("\n")
        f.close()
        
    def _dump_grades(self, filename, grades):
        """Dump the specified feature matrix to human-readable file."""
        f = open(filename, 'w')
        for grade in grades:
            f.write("%d\n" %grade)
        f.close()
        
    def dump(self):
        """Dump the essays and grades to human-readable file."""
        self._dump_feat_mat("output/features.set%d.train" %self.ds_train.getEssaySet(), self.train_feat_mat)
        self._dump_feat_mat("output/features.set%d.test" %self.ds_test.getEssaySet(), self.test_feat_mat)

        self._dump_grades("output/grades.set%d.train" %self.ds_train.getEssaySet(), self.ds_train.getGrades())
        self._dump_grades("output/grades.set%d.test" %self.ds_test.getEssaySet(), self.ds_test.getGrades())





























