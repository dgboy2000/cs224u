import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pylab as P



class ErrorAnalysis:
    """Represents an error analysis."""
    def __init__(self, essay_set, grade_domain, test=True):
        self.essay_set = essay_set
        self.grade_domain = grade_domain
        self.is_test_set = test
        
        self.essay_filename = "output/diffs.set%d.domain%d.%s" %(essay_set, grade_domain, "test" if test else "train")
        self.grade_filename = None
        
        self._read_essay_tsv_file(self.essay_filename)
        self._read_grade_tsv_file(self.grade_filename)
        
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
                            datamap[rowmap[i]].append(float(col))
                        else:
                            datamap[rowmap[i]].append('**GARBAGE**')
                    i += 1

        self.rowmap = rowmap
        self.datamap = datamap
        
    def _read_grade_tsv_file(self, grade_filename):
        pass

    def all_grade_errors_by_count(self, bins=10):
        errors = [self.datamap['pred_grade'][essay_ind] - gt_grade for essay_ind, gt_grade in enumerate(self.datamap['gt_grade'])]

        plt.clf()
        fig = plt.figure()        
        ax = fig.add_subplot(111)
        ax.hist(errors, bins)
        
        plt.savefig('output/all_grade_errors_by_count.set%d.domain%d.%s.png' %(self.essay_set, self.grade_domain, "test" if self.is_test_set else "train"), format='png')
        
    def all_score_errors_by_count(self, bins=10):
        errors = np.asarray([self.datamap['pred_score'][essay_ind] - gt_grade for essay_ind, gt_grade in enumerate(self.datamap['gt_grade'])])
        mu = np.mean(errors)
        var = np.var(errors)
        sd = np.sqrt(var)

        plt.clf()
        fig = plt.figure()        
        ax = fig.add_subplot(111)
        pdf, bins, patches = ax.hist(errors, bins)

        normal_dist = [len(errors) * (bins[-1]-bins[0]) / (len(bins)-1) * 1/(sd*np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/(2*var)) for x in bins]
        ax.plot(bins, normal_dist)

        plt.savefig('output/all_score_errors_by_count.set%d.domain%d.%s.png' %(self.essay_set, self.grade_domain, "test" if self.is_test_set else "train"), format='png')
            
            

ea = ErrorAnalysis(1, 1, test=True)
ea.all_grade_errors_by_count(bins=20)
ea.all_score_errors_by_count(bins=40)










































