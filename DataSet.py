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

class DataSet:
    def __init__(self):
        self.colNames = list()
        self.rows = list() # dataset is a matrix - one level deep of nested lists
        self.textOnly = list() # just a list of the text
        self.isTrainSet = False
        self.domain_id = 1
        self.grades = list()
        self.prediction_ids = list()
        self.essay_ids = list()
        self.essay_set = None
        self.file_id = 'default'

    def getID(self):
        return self.file_id

    def setID(self, file_id):
        self.file_id = file_id

    def importData(self, filename, essay_set=-1, domain_id=1):
        """If essay_set=-1, then we use all essays."""
        reader = csv.reader(open(filename, 'rb'), delimiter='\t', quotechar='"')
        first = True

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
                    self.textOnly.append(row[text_col])
                    self.essay_ids.append(int(row[essay_id_col]))

                    if grade_col > -1:
                        self.grades.append(int(row[grade_col]))
                    if pred_id_col > -1:
                        self.prediction_ids.append(int(row[pred_id_col]))

        return

    def setTrainSet(self, val):
        self.isTrainSet = val
        return

    def size(self):
        return len(self.rows)

    def dumpColNames(self):
        print '\n'.join(self.colNames)

    def dumpRow(self, lineNum):
        print '\n'.join(self.rows[lineNum])

    def getRawText(self):
        return self.textOnly

    def getEssaySet(self):
        return self.essay_set

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
