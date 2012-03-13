import csv
import nltk
import os
import pickle
import re
from spelling.SpellChecker import SpellChecker

class CreateDictionary:

    word_splitter = nltk.WordPunctTokenizer()
    spell_checker = SpellChecker()
    dictionary_fn = 'cache/dictionary.pkl'
    suggestions_fn = 'cache/spell_suggestions.pkl'

    def __init__(self):
        if not os.path.exists(CreateDictionary.dictionary_fn):
            self.extractCounts()

    def getDictionary(self):
        return pickle.load(open(CreateDictionary.dictionary_fn, 'r'))
    
    def getSpellingSuggestions(self):
        return pickle.load(open(CreateDictionary.suggestions_fn, 'r'))

    def addValidWords(self, essay, word_list):
        newWords = [word.lower() for word in CreateDictionary.word_splitter.tokenize(essay) if (word.isalpha() or re.match("^[a-zA-Z][a-zA-Z']*$", word))]
        for word in newWords:
            word_list.append(word)

    def extractCounts(self):
        word_list = list()
        dictionary = set()
        suggestions = {}
        
        reader = csv.reader(open('data/c_train.utf8ignore.tsv', 'r'), delimiter='\t')
        for row in reader:
            self.addValidWords(row[2], word_list)

        reader = csv.reader(open('data/c_val.utf8ignore.tsv', 'r'), delimiter='\t')
        for row in reader:
            self.addValidWords(row[2], word_list)

        reader = csv.reader(open('data/valid_set.utf8ignore.tsv', 'r'), delimiter='\t')
        for row in reader:
            self.addValidWords(row[2], word_list)

        word_set = set(word_list)
        word_set.discard('')

        for word in word_set:
            result = CreateDictionary.spell_checker.extractSpellingSuggestions(word)
            if result is None:
                dictionary.add(word)
                
                if len(dictionary)%100 is 0:
                    print ('dictionary word count = ' + str(len(dictionary)))
            else:
                suggestions[word] = result

        dictionary.add('facebook')

        pickle.dump(dictionary, open(CreateDictionary.dictionary_fn, 'w'))
        pickle.dump(suggestions, open(CreateDictionary.suggestions_fn, 'w'))
