import DataSet
import nltk
import os
import pickle
from spelling.CreateDictionary import CreateDictionary

class WordCounter:

    word_counter_fn = 'cache/word_counter.pkl'
    word_splitter = nltk.WordPunctTokenizer()

    def __init__(self):
        if not os.path.exists(WordCounter.word_counter_fn):
            self.extractCounts()

    def getCounts(self):
        return pickle.load(open(WordCounter.word_counter_fn, 'r'))

    def addValidWords(self, ds, word_list):
        for line in ds.getRawText():            
            newWords = [word.lower() for word in WordCounter.word_splitter.tokenize(line) if word.isalpha()]
            for word in newWords:
                word_list.append(word)

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

            self.addValidWords(ds_train, word_list)
            self.addValidWords(ds_val, word_list)

        print 'importing real validation set'
        ds_val = DataSet.DataSet()
        ds_val.importData('data/valid_set.tsv')
        ds_val.setTrainSet(False)
        self.addValidWords(ds_val, word_list)

        pickle.dump(word_counter, open(WordCounter.word_counter_fn, 'w'))
