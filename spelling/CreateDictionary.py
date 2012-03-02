import DataSet
import nltk
import os
import pickle
from spelling.SpellChecker import SpellChecker

class CreateDictionary:

    word_splitter = nltk.WordPunctTokenizer()
    spell_checker = SpellChecker()
    dictionary_fn = 'cache/dictionary.pkl'

    def __init__(self):
        if not os.path.exists(CreateDictionary.dictionary_fn):
            self.extractCounts();

    def getDictionary(self):
        return pickle.load(open(CreateDictionary.dictionary_fn))

    def addValidWords(self, ds, word_list):
        for line in ds.getRawText():
            newWords = [word.lower() for word in CreateDictionary.word_splitter.tokenize(line) if word.isalpha()]
            for word in newWords:
                word_list.append(word)

    def extractCounts(self):
        word_list = list()
        dictionary = set()

        for essay_set in range(1, 9): 
            ds_train = DataSet.DataSet()
            ds_train.importData('data/c_train.tsv', essay_set)
            ds_train.setTrainSet(True)
            self.addValidWords(ds_train, word_list);

            ds_val = DataSet.DataSet()
            ds_val.importData('data/c_val.tsv', essay_set)
            ds_val.setTrainSet(False)
            self.addValidWords(ds_val, word_list);
            
        ds_val = DataSet.DataSet()
        ds_val.importData('data/valid_set.tsv')
        ds_val.setTrainSet(False)
        self.addValidWords(ds_val, word_list)

        word_set = set(word_list)
        word_set.discard('')

        for word in word_set:
            if CreateDictionary.spell_checker.extractSpellingSuggestions(word) is None:
                dictionary.add(word)
                if len(dictionary)%100 is 0:
                    print len(dictionary)

        pickle.dump(dictionary, open(CreateDictionary.dictionary_fn))
