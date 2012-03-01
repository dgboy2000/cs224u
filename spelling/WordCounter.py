import DataSet
import nltk
import os
import pickle
from spelling.CreateDictionary import CreateDictionary

class WordCounter:

    word_counter_fn = 'cache/word_counter.pkl'
    word_splitter = nltk.WordPunctTokenizer()

    def __init__(self):
        if not os.path.exists(word_counter_fn)
            extractCounts()

    def getCounts(self):
        return pickle.load(open(word_counter_fn, 'r'))

    def extractCounts(self):
        dic_obj = CreateDictionary()
        dictionary = dic_obj.getDictionary()

        word_counter = {}

        for word in dictionary:
            word_counter[word] = 0

        for essay_set in range(1, 9):
            print 'importing train set'
            ds_train = DataSet.DataSet()
            ds_train.importData('data/c_train.tsv', essay_set)
            ds_train.setTrainSet(True)
            print 'importing val set'
            ds_val = DataSet.DataSet()
            ds_val.importData('data/c_val.tsv', essay_set)
            ds_val.setTrainSet(False)

            for line in ds_train.getRawText():
                for word in word_splitter.tokenize(line):
                    if word in dictionary:
                        word_counter[word] += 1

            for line in ds_val.getRawText():        
                for word in word_splitter.tokenize(line):
                    if word in dictionary:
                        word_counter[word] += 1

        print 'importing real validation set'
        ds_val = DataSet.DataSet()
        ds_val.importData('data/valid_set.tsv')
        ds_val.setTrainSet(False)

        for line in ds_val.getRawText():        
            for word in word_splitter.tokenize(line):
                if word in dictionary:
                    word_counter[word] += 1

        pickle.dump(word_counter, open(word_counter_fn, 'w'))
