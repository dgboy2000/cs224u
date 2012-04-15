import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pylab as P



class ErrorAnalysis:
    """Represents an error analysis."""
    def __init__(self, essay_filename, grade_filename):
        self._read_essay_tsv_file(essay_filename)
        self._read_grade_tsv_file(grade_filename)
        
    def _read_essay_tsv_file(self, essay_filename):
        reader = csv.reader(open(essay_filename, 'rb'), delimiter='\t', quotechar=None)
        first = True

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
                    elif rowmap[i] in ['gt_grade', 'pred_score', 'pred_grade']:
                        datamap[rowmap[i]].append(float(col))
                    else:
                        if col:
                            datamap[rowmap[i]].append(int(col))
                        else:
                            datamap[rowmap[i]].append('**GARBAGE**')
                    i += 1

        self.rowmap = rowmap
        self.datamap = datamap

    def all_errors_by_count(self, bins=10):
        errors = [self.datamap['pred_grade'][essay_ind] - gt_grade for essay_ind, gt_grade in enumerate(self.datamap['gt_grade'])]

        plt.clf()
        fig = plt.figure()        
        ax = fig.add_subplot(111)
        ax.hist(errors, bins)
        
        plt.savefig('all_errors_by_count.png', format='png')
        


ea = ErrorAnalysis('output')










































