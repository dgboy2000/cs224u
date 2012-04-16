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
        self.feature_filename = "output/features.set%d.dom%d.%s" %(essay_set, grade_domain,"test" if test else "train")
        self.features = None
        
        self._read_essay_tsv_file(self.essay_filename)
        self._read_grade_tsv_file(self.grade_filename)
        self._read_feature_tsv_file(self.feature_filename)
        
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

    def _read_feature_tsv_file(self, feature_filename):
        reader = csv.reader(open(feature_filename, 'rb'), delimiter='\t', quotechar=None)
        #first = True #No header row in feature files
        
        #rowmap = dict() # key = col index, value = col_header_name
        datamap = dict() # key = col_header_name, value = list of data
        #import pdb
        #pdb.set_trace()
        row1 = reader.next()
        rowlen = len(row1)-1
        row_num = 1
        for i in range(rowlen): # read and count first row
            datamap[i] = list()
            try:
                datamap[i].append(float(row1[i]))
            except:
                import pdb; pdb.set_trace()
        for row in reader: #read all other rows
            row_num += 1
            for i in range(rowlen):
                try:
                    datamap[i].append(float(row[i]))
                except:
                    import pdb; pdb.set_trace()
        self.features = datamap
          

    def all_grade_errors_by_count(self, bins=10):
        errors = [self.datamap['pred_grade'][essay_ind] - gt_grade for essay_ind, gt_grade in enumerate(self.datamap['gt_grade'])]

        plt.clf()
        fig = plt.figure()        
        ax = fig.add_subplot(111)
        ax.hist(errors, bins)
        ax.set_title("All %s Grade Errors, Essay Set %d, Domain %d" %("test" if self.is_test_set else "train", self.essay_set, self.grade_domain))
        ax.set_xlabel("Predicted Grade - Actual Grade")
        ax.set_ylabel("Number of Occurrences")
        
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
        ax.set_title("All %s Score Errors, Essay Set %d, Domain %d" %("test" if self.is_test_set else "train", self.essay_set, self.grade_domain))
        ax.set_xlabel("Predicted Score - Actual Grade")
        ax.set_ylabel("Number of Occurrences")

        plt.savefig('output/all_score_errors_by_count.set%d.domain%d.%s.png' %(self.essay_set, self.grade_domain, "test" if self.is_test_set else "train"), format='png')
            
    def mean_score_error_by_grade(self):
        grade_to_errors = {}
        grade_to_mean_error = {}

        for essay_ind, gt_grade in enumerate(self.datamap['gt_grade']):
            if gt_grade not in grade_to_errors:
                grade_to_errors[gt_grade] = []
            grade_to_errors[gt_grade].append(self.datamap['pred_score'][essay_ind] - gt_grade)
        left = []
        height = []
        for grade, errors in grade_to_errors.iteritems():
            grade_to_mean_error[grade] = np.mean(np.abs(errors))
            left.append(grade - 0.5)
            height.append(np.mean(np.abs(errors)))
        
        
        plt.clf()
        fig = plt.figure()        
        ax = fig.add_subplot(111)
        ax.bar(left, height, width = 1)
        ax.set_title("Mean %s Score Error by Grade, Essay Set %d, Domain %d" %("test" if self.is_test_set else "train", self.essay_set, self.grade_domain))
        ax.set_xlabel("Actual Grade")
        ax.set_ylabel("Mean(Absolute Value of Predicted Score - Actual Grade)")
        
        plt.savefig('output/mean_score_error_by_grade.set%d.domain%d.%s.png' %(self.essay_set, self.grade_domain, "test" if self.is_test_set else "train"), format='png')
        
    def scatter_errors_by_grade(self):
        errors = np.asarray([self.datamap['pred_score'][essay_ind] - gt_grade for essay_ind, gt_grade in enumerate(self.datamap['gt_grade'])])
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(self.datamap['gt_grade'], errors, marker='o', alpha = 0.25)
        ax.set_title("%s Error by Actual Grade, Essay Set % d, Domain %d " %("Test" if self.is_test_set else "train", self.essay_set, self.grade_domain) )
        ax.set_xlabel("Actual Grade")
        ax.set_ylabel("Predicted Score - Actual Grade")
        plt.savefig('output/scatter_errors_by_grade.set%d.domain%d.%s.png' %(self.essay_set, self.grade_domain, "test" if self.is_test_set else "train"), format='png')
          
    def score_errors_by_grade(self, bins=10):
        grades = np.asarray(self.datamap['gt_grade'])
        min_grade = int(min(grades))
        max_grade = int(max(grades))
        for g in range(min_grade, max_grade+1):
            curr_inds = np.asarray(np.where(grades==g)[0]) #indices of current grade in grades[]
            if(len(curr_inds)==0): continue
            errors = np.asarray([self.datamap['pred_score'][essay_ind] - gt_grade for essay_ind, gt_grade in enumerate(self.datamap['gt_grade'])])
            curr_errors = []
            for ind in curr_inds:
                curr_errors.append(errors[ind])
            curr_errors = np.asarray(curr_errors)
            mu = np.mean(curr_errors)
            var = np.var(curr_errors)
            sd = np.sqrt(var)

            plt.clf() 
            fig = plt.figure()         
            ax = fig.add_subplot(111)
            try:
                pdf, bins, patches = ax.hist(curr_errors, bins)
            except:
                 import pdb; pdb.set_trace()
   
            normal_dist = [len(curr_errors) * (bins[-1]-bins[0])/(len(bins)-1) * 1/(sd*np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/(2*var)) for x in bins] 
            ax.plot(bins, normal_dist)
            ax.set_title("All %s Errors by Grade, Essay Set %d, Domain %d" %("test" if self.is_test_set else "train", self.essay_set, self.grade_domain)) 
            ax.set_xlabel("Actual Grade") 
            ax.set_ylabel("Predicted Score - Actual Grade")

            plt.savefig('output/all_score_errors_by_count.set%d.domain%d.grade%d.%s.png' %(self.essay_set, self.grade_domain, g, "test" if self.is_test_set else "train"), format='png')

    def count_errors_by_grade(self):
        grades = np.asarray(self.datamap['gt_grade'])
        min_grade = int(min(grades))
        max_grade = int(max(grades))
        x = []
        y = []
        for g in range(min_grade, max_grade+1):
            curr_inds = np.asarray(np.where(grades==g)[0]) #indices of current grade in grades[]
            if(len(curr_inds)==0): continue
            x.append(g)
            errors = np.asarray([self.datamap['pred_grade'][essay_ind] - gt_grade for essay_ind, gt_grade in enumerate(self.datamap['gt_grade'])])
            count = 0
            for ind in curr_inds:
                if errors[ind] != 0: count += 1
            y.append(count)
        #import pdb; pdb.set_trace()
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x, y)
        ax.set_title("Count of %s Grade Errors, Essay Set %d, Domain %d" %("Test" if self.is_test_set else "Train", self.essay_set, self.grade_domain))
        ax.set_ylabel("Count of Errors in Predicted Grades")
        ax.set_xlabel("Actual Grade")
        plt.savefig('output/count_of_grade_errors_by_grade.set%d.domain%d.%s.png' %(self.essay_set, self.grade_domain, "test" if self.is_test_set else "train"), format='png')

    def plot_resid_by_feature(self, set, domain, features_used):
        features = self.features
        errors = [self.datamap['pred_score'][essay_ind] - gt_grade for essay_ind, gt_grade in enumerate(self.datamap['gt_grade'])]
        #import pdb; pdb.set_trace()        
        for f in features_used:
            feature_values = features[f]
            plt.clf() 
            fig = plt.figure()         
            ax = fig.add_subplot(111)
            ax.scatter(feature_values, errors, marker='o', alpha = 0.25) 
            ax.set_title("All %s Errors, Essay Set %d, Domain %d, Feature = %d" %("test" if self.is_test_set else "train", self.essay_set, self.grade_domain, f))
            ax.set_ylabel("Predicted Score - Actual Grade")
            ax.set_xlabel("Values of Feature %d" %(f))

            plt.savefig('output/by_feature/all_errors_by_feature.set%d.domain%d.feature%d.%s.png' %(self.essay_set, self.grade_domain, f, "test" if self.is_test_set else "train"), format='png')
            
    def plot_feature_by_gtgrade(self, set, domain, features_used):
        features = self.features
        #errors = [self.datamap['pred_grade'][essay_ind] - gt_grade for essay_ind, gt_grade in enumerate(self.datamap['gt_grade'])]
        grades = np.asarray(self.datamap['gt_grade'])
        for f in features_used:
            feature_values = features[f]
            plt.clf() 
            fig = plt.figure()         
            ax = fig.add_subplot(111)
            ax.scatter(feature_values, self.datamap['gt_grade'], marker='o', alpha = 0.25) 
            ax.set_title("Feature = %d vs. %s GT Grade, Essay Set %d, Domain %d, " %(f, "test" if self.is_test_set else "train", self.essay_set, self.grade_domain))
            ax.set_ylabel("Actual Grade")
            ax.set_xlabel("Values of Feature %d" %(f))

            plt.savefig('output/by_feature/gt_grade_by_feature.set%d.domain%d.feature%d.%s.png' %(self.essay_set, self.grade_domain, f, "test" if self.is_test_set else "train"), format='png')
    
if __name__ == '__main__':
    
    features_used = dict()
    features_used[1] = [0,1,2,3,42,44]
    features_used[2] = [0,2,3,8,11,14,23,27,43,44,52]
    features_used[3] = [2,7,22,43,48]
    
    for essay_set in range(1,9):
        print "Analyzing errors for essay set %d, domain 1..." %essay_set,
        ea = ErrorAnalysis(essay_set, 1, test=True)
        for set in [1,2,3]:
            if set == essay_set:
                ea.plot_resid_by_feature(set, 1, features_used[set])
                ea.plot_feature_by_gtgrade(set, 1, features_used[set])

        ea.all_grade_errors_by_count(bins=20)
        ea.all_score_errors_by_count(bins=40)
        ea.mean_score_error_by_grade()
        ea.scatter_errors_by_grade()
        ea.score_errors_by_grade(bins=30)
        ea.count_errors_by_grade()
        print "Done"
        
        if essay_set == 2:
            
            print "Analyzing errors for essay set %d, domain 2..." %essay_set,
            ea = ErrorAnalysis(essay_set, 2, test=True)
            #ea.plot_resid_by_feature(2, 2, features_used[2])
            
            ea.all_grade_errors_by_count(bins=20)
            ea.all_score_errors_by_count(bins=40)
            ea.mean_score_error_by_grade()
            ea.scatter_errors_by_grade()
            ea.score_errors_by_grade(bins=30)
            ea.count_errors_by_grade()
            ea.plot_resid_by_feature(2,2,features_used[2])
            ea.plot_feature_by_gtgrade(2,2,features_used[2])
            print "Done"


     
































