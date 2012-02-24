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

class DataSet:
    def __init__(self):
        self.colNames = list()
        self.rows = list() # dataset is a matrix - one level deep of nested lists
        self.textOnly = list() # just a list of the text
        self.isTrainSet = False

    def importData(self, filename):
        reader = csv.reader(open(filename, 'rb'), delimiter='\t', quotechar='"')
        first = True

        for row in reader:
            if first:
                self.colNames = row
                first = False
            else:
                self.rows.append(row)
                self.textOnly.append(row[2])

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

    def getAggGrade(self, lineNum):
        pass
