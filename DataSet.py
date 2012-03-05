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

class DataSet:
    def __init__(self):
        self.colNames = list()
        self.rows = list() # dataset is a matrix - one level deep of nested lists
        self.textOnly = list() # just a list of the text
        self.trainSetFlag = False
        self.domain_id = 1
        self.grades = list()
        self.prediction_ids = list()
        self.essay_ids = list()
        self.essay_set = None
        self.file_id = 'default'
        self.file_name = ''
        self.pos_tags = list()
        self.gensim_corpus = ()

    def getID(self):
        return self.file_id

    def setID(self, file_id):
        self.file_id = file_id

    def getFilename(self):
        return self.file_name

    def importData(self, filename, essay_set=-1, domain_id=1):
        """If essay_set=-1, then we use all essays."""
        reader = csv.reader(open(filename, 'rb'), delimiter='\t', quotechar=None)
        first = True
        self.file_name = os.path.basename(filename)

        self.domain_id = domain_id
        self.essay_set = essay_set

        grade_col = -1
        text_col = -1
        essay_set_col = -1
        pred_id_col = -1
        essay_id_col = -1

        for row in reader:
            if first:
                self.colNames = row
                i = 0
                for col in row:
                    if col == 'essay_set':
                        essay_set_col = i
                    elif col == 'essay':
                        text_col = i
                    elif col == 'essay_id':
                        essay_id_col = i
                    elif domain_id == 1 and col == 'domain1_score':
                        grade_col = i
                    elif domain_id == 2 and col == 'domain2_score':
                        grade_col = i
                    elif domain_id == 1 and col == 'domain1_predictionid':
                        pred_id_col = i
                    elif domain_id == 2 and col == 'domain2_predictionid':
                        pred_id_col = i
                    i += 1
                first = False
            else:
                if essay_set < 0 or int(row[essay_set_col]) == essay_set:
                    self.rows.append(row)
                    self.textOnly.append(row[text_col].strip('"'))
                    self.essay_ids.append(int(row[essay_id_col]))

                    if grade_col > -1:
                        self.grades.append(int(row[grade_col]))
                    if pred_id_col > -1:
                        self.prediction_ids.append(int(row[pred_id_col]))

        return

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

    def setTrainSet(self, val):
        self.trainSetFlag = val
        return

    def isTrainSet(self):
        return self.trainSetFlag

    def size(self):
        return len(self.rows)

    def dumpColNames(self):
        print '\n'.join(self.colNames)

    def dumpRow(self, lineNum):
        print '\n'.join(self.rows[lineNum])

    def setGensimCorpus(self, mm):
        self.gensim_corpus = mm

    def getGensimCorpus(self):
        return self.gensim_corpus

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
