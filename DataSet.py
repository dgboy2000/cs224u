# Defines a dataset.
# This class defines helper functions for datasets, such as:
#   1) number of lines
#   2) line by line data
#   3) getting class information
#   4) in the future: n-folds

# An object of this type should be fed directly into feature-grabbing functions
# TODO: should we feed it directly into a learning / inference function?
#       that means we'd hold features in this class.

import csv
import numpy as np
import os
import LanguageUtils
import nltk
import cPickle as pickle
from nltk.collocations import *

# OTHER_DISTS lists the other grade distributions that each essay set uses, *other* than the resolved grade.
OTHER_DISTS = {
    (1, 1): ['rater1_domain1', 'rater2_domain1'],
    (2, 1): ['rater1_domain1', 'rater2_domain1', 'domain2_score', 'rater1_domain2', 'rater2_domain2'],
    (2, 2): ['rater1_domain1', 'rater2_domain1', 'domain1_score', 'rater1_domain2', 'rater2_domain2'],
    (3, 1): ['rater1_domain1', 'rater2_domain1'],
    (4, 1): ['rater1_domain1', 'rater2_domain1'],
    (5, 1): ['rater1_domain1', 'rater2_domain1'],
    (6, 1): ['rater1_domain1', 'rater2_domain1'],
    (7, 1): ['rater1_domain1', 'rater2_domain1'],
    (8, 1): ['rater1_domain1', 'rater2_domain1'], #, 'rater3_domain1'], <- rater3 is incomplete
    }

class DataSet:
    def __init__(self, trainSetFlag):
        self.colNames = list()
        self.ds_size = -1
        self.textOnly = list() # just a list of the text
        self.trainSetFlag = trainSetFlag
        self.domain_id = 1
        self.grades = list()
        self.prediction_ids = list()
        self.essay_ids = list()
        self.essay_set = None
        self.file_name = ''
        self.pos_tags = list()
        self.bigram_pos_tags = list()
        self.trigram_pos_tags = list()
        self.gensim_corpus = ()
        self.other_dists_grades = list()

    def getFilename(self):
        return self.file_name

    def importData(self, filename, essay_set=-1, domain_id=1):
        """If essay_set=-1, then we use all essays."""

        if domain_id != 1 and domain_id != 2:
            raise Exception("Unknown Domain.")

        reader = csv.reader(open(filename, 'rb'), delimiter='\t', quotechar=None)
        first = True
        self.file_name = os.path.basename(filename)

        self.domain_id = domain_id
        self.essay_set = essay_set

        rowmap = dict() # key = col index, value = col_header_name
        datamap = dict() # key = col_header_name, value = list of data
        for row in reader:
            if first:
                i = 0
                for col in row:
                    rowmap[i] = col
                    datamap[col] = list()
                    i += 1

                first = False
            else:
                i = 0
                for col in row:
                    if rowmap[i] == 'essay':
                        datamap[rowmap[i]].append(col.strip('"'))
                    else:
                        if col:
                            datamap[rowmap[i]].append(int(col))
                        else:
                            datamap[rowmap[i]].append('**GARBAGE**')
                    i += 1

        # get indices that match essay_set
        inds = [i for i in range(len(datamap['essay_set'])) 
                if datamap['essay_set'][i] == self.getEssaySet() or self.getEssaySet() == -1]
        self.ds_size = len(inds)

        # get master grades
        domain_col_name = 'domain%d_score' % self.getDomain()
        if domain_col_name in datamap:
            self.grades = [datamap[domain_col_name][i] for i in inds]

        # get textOnly
        self.textOnly = [datamap['essay'][i] for i in inds]

        # get essay_ids
        self.essay_ids = [datamap['essay_id'][i] for i in inds]

        # get prediction_ids
        pred_col_name = 'domain%d_predictionid' % self.getDomain()
        if pred_col_name in datamap:
            self.prediction_ids = [datamap[pred_col_name][i] for i in inds]

        # get other_dists
        if (self.isTrainSet()):
            other_dists_cols = OTHER_DISTS[(self.getEssaySet(), self.getDomain())]
            self.other_dists_grades = list()
            for dist_col_name in other_dists_cols:
                self.other_dists_grades.append([datamap[dist_col_name][i] for i in inds])
        else:
            self.other_dists_grades = list()

        return

    def getOtherDistsGrades(self):
        return self.other_dists_grades

    def getAllBoW(self):
        """Return unigram, bigram words as bag of words."""

        fname = 'cache/word_ngrams.%s.set%d.dom%d.pickle' % (
                 self.getFilename(), self.getEssaySet(), self.getDomain())

        try:
            f = open(fname, 'rb')
            bow = pickle.load(f)
        except:
            bigram_measures = nltk.collocations.BigramAssocMeasures()

            bow = list()
            for line in self.getRawText():
                cur = LanguageUtils.tokenize(line)

                finder = BigramCollocationFinder.from_words(cur)
                scored = finder.score_ngrams(bigram_measures.pmi)
                for bigram, score in scored:
                    cur.append(bigram)

                bow.append(cur)

            pickle.dump(bow, open(fname, 'w'))

        return bow

    def getAllPOS(self):
        """Return unigram, bigram, trigram POS tags as bag of words."""

        fname = 'cache/pos_ngrams.%s.set%d.dom%d.pickle' % ( 
                 self.getFilename(), self.getEssaySet(), self.getDomain())
        try:
            f = open(fname, 'rb')
            all = pickle.load(f)
        except:
            uni = self.getPOS()
            bi = self.getBigramPOS()
            tri = self.getTrigramPOS()
            all = list()
            for i in range(self.size()):
                all.append(uni[i] + bi[i] + tri[i])

            pickle.dump(all, open(fname, 'w'))

        return all

    def getPOS(self):
        if len(self.pos_tags) > 0:
            return self.pos_tags

        fname = 'cache/pos.%s.set%d.pickle' % (self.file_name, self.essay_set)
        try:
            f = open(fname, 'rb')
            self.pos_tags = pickle.load(f)
        except:
            pos_lines = list()
            tot_ln = self.size()
            prog = 0
            hunpos = nltk.tag.HunposTagger("en_wsj.model")

            for line in self.getRawText():
                tokens = LanguageUtils.punkt_tokenize(line)
                pos_tags = hunpos.tag(tokens)
                tags_only = [tag for w, tag in pos_tags]
                pos_lines.append(tags_only)
                prog += 1
                if prog % 100 == 0:
                    print "POS Tagging %d of %d" % (prog, tot_ln)

                self.pos_tags = pos_lines

            f = open(fname, 'w')
            pickle.dump(self.pos_tags, f)

        return self.pos_tags

    def getBigramPOS(self):
        if len(self.bigram_pos_tags) > 0:
            return self.bigram_pos_tags

        bigram_measures = nltk.collocations.BigramAssocMeasures()

        bigram_tags = list()
        for tokens in self.pos_tags:
            finder = BigramCollocationFinder.from_words(tokens)
            cur_scored = finder.score_ngrams(bigram_measures.pmi)

            bigrams = list()
            for bigram, score in cur_scored:
                bigrams.append(bigram)

            bigram_tags.append(bigrams)

        self.bigram_pos_tags = bigram_tags

        return self.bigram_pos_tags

    def getTrigramPOS(self):
        if len(self.trigram_pos_tags) > 0:
            return self.trigram_pos_tags

        trigram_measures = nltk.collocations.TrigramAssocMeasures()

        trigram_tags = list()
        for tokens in self.pos_tags:
            finder = TrigramCollocationFinder.from_words(tokens)
            cur_scored = finder.score_ngrams(trigram_measures.pmi)

            trigrams = list()
            for trigram, score in cur_scored:
                trigrams.append(trigram)

            trigram_tags.append(trigrams)

        self.trigram_pos_tags = trigram_tags

        return self.trigram_pos_tags

    def setTrainSet(self, val):
        self.trainSetFlag = val
        return

    def isTrainSet(self):
        return self.trainSetFlag

    def size(self):
        return self.ds_size

    def dumpColNames(self):
        print '\n'.join(self.colNames)

    def setGensimCorpus(self, mm):
        self.gensim_corpus = mm

    def getGensimCorpus(self):
        return self.gensim_corpus

    def setGensimPOSCorpus(self, mm):
        self.gensim_pos_corpus = mm

    def getGensimPOSCorpus(self):
        return self.gensim_pos_corpus

    def getRawText(self):
        return self.textOnly

    def getEssaySet(self):
        return self.essay_set

    def getDomain(self):
        return self.domain_id

    def outputKaggle(self, grades, fd):
        """Output standard Kaggle validation set format. file will be appended to."""

        predweight = '1'
        if self.essay_set == 2:
            predweight = '0.5'

        for i in range(self.size()):
            str = "%d\t%d\t%d\t%s\t%d\n" % (self.prediction_ids[i], self.essay_ids[i], self.essay_set, predweight, grades[i])
            fd.write(str)

    # Returns a numpy array of the rades
    def getGrades(self):
        # TODO throw exception if grades is empty
        return np.asarray(self.grades)
        
    def setGrades(self, grades):
        self.grades = grades
        


















