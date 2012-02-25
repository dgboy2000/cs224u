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

class DataSet:
    def __init__(self):
        self.colNames = list()
        self.rows = list() # dataset is a matrix - one level deep of nested lists
        self.textOnly = list() # just a list of the text
        self.isTrainSet = False
        self.grades = list()

    def importData(self, filename):
        reader = csv.reader(open(filename, 'rb'), delimiter='\t', quotechar='"')
        first = True

        for row in reader:
            if first:
                self.colNames = row
                first = False
                print row[6]
                print row[9]
            else:
                self.rows.append(row)
                self.textOnly.append(row[2])

                if (row[6]):
                    self.grades.append(int(row[6]))
                else:
                    self.grades.append(int(row[9]))

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

    # Returns a numpy array of the rades
    def getGrades(self):
        return np.asarray(self.grades)
