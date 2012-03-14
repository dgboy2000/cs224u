from spelling.CreateDictionary import CreateDictionary
import csv
import math
import nltk
import os
import pickle
import string
import sys

class SpellCorrector:

    TOTAL_COUNT = '**total**'
   
    EDIT_WEIGHT = -3
    UNIGRAM_WEIGHT = 1
    BIGRAM_WEIGHT = 1
    
    word_splitter = nltk.WordPunctTokenizer()

    dic_obj = CreateDictionary()
    dictionary = dic_obj.getDictionary()
    dictionary.add('.')
    dictionary.add(',')
    dictionary.add('?')
    dictionary.add('!')
    dictionary.add('(')
    dictionary.add(')')
    dictionary.add(':')
    dictionary.add(';')
    dictionary.add('-')
    dictionary.add('&')
    suggestions = dic_obj.getSpellingSuggestions() 

    unigrams_fn = 'cache/uk_unigrams.pkl'
    bigrams_fn = 'cache/uk_bigrams.pkl'
    unigrams = {}
    bigrams = {}

    def __init__(self):
        if not os.path.exists(SpellCorrector.unigrams_fn):
            self.readUnigrams()
        else:
            SpellCorrector.unigrams = pickle.load(open(SpellCorrector.unigrams_fn, 'rb'))
        if not os.path.exists(SpellCorrector.bigrams_fn):
            self.readBigrams()
        else:
            SpellCorrector.bigrams = pickle.load(open(SpellCorrector.bigrams_fn, 'rb'))

    def readUnigrams(self):
        csv.field_size_limit(sys.maxint)
        reader = csv.reader(open('data/uk_unigrams.txt', 'r'), delimiter='\t')
        total_count = 0

        for row in reader:
            # dividing counts by 1000 keeps the numbers smaller and allows us
            # to filter out rare/incorrect words
            count = int(row[0]) / 200
            if count > 0:
                word = string.lower(row[1])
                if word in SpellCorrector.unigrams:
                    SpellCorrector.unigrams[word] += count
                else:
                    SpellCorrector.unigrams[word] = count
                total_count += count

        SpellCorrector.unigrams[SpellCorrector.TOTAL_COUNT] = total_count
        pickle.dump(SpellCorrector.unigrams, open(SpellCorrector.unigrams_fn, 'wb'))

    def readBigrams(self):
        csv.field_size_limit(sys.maxint)
        reader = csv.reader(open('data/uk_bigrams.txt', 'r'), delimiter='\t')
        total_count = 0

        for row in reader:
            count = int(row[0]) / 200
            if count > 0:
                bigram = string.lower(row[1] + ' ' + row[2])
                if bigram in SpellCorrector.bigrams:
                    SpellCorrector.bigrams[bigram] += count
                else:
                    SpellCorrector.bigrams[bigram] = count
                total_count += count
        
        SpellCorrector.bigrams[SpellCorrector.TOTAL_COUNT] = total_count
        pickle.dump(SpellCorrector.bigrams, open(SpellCorrector.bigrams_fn, 'wb'))

    def correctEssaySets(self, filename_in, filename_out):
        reader = csv.reader(open(filename_in, 'r'), delimiter = '\t')
        output_file = open(filename_out, 'w')

        for row in reader:
            essay = row[2].strip('"')
            output_file.write(self.correctEssay(essay) + '\n')
        output_file.close() 

    def correctEssay(self, essay):
        words = [word for word in SpellCorrector.word_splitter.tokenize(essay)]
        if len(words) < 3:
            return essay

        corrected_essay = self.correctWord(None, words[0], words[1]) + ' '    

        for i in range(1, len(words)-1):
            corrected_essay += self.correctWord(words[i-1], words[i], words[i+1]) + ' '    
        corrected_essay += self.correctWord(words[len(words)-2], words[len(words)-1], None)    
        
        corrected_essay = corrected_essay.replace("' ", "'");
        corrected_essay = corrected_essay.replace(" n't", "n't");
        corrected_essay = corrected_essay.replace(" '", "'");
        corrected_essay = corrected_essay.replace('@ ', '@');
        corrected_essay = corrected_essay.replace(' !', '!');
        corrected_essay = corrected_essay.replace(' ?', '?');
        corrected_essay = corrected_essay.replace(' ,', ',');
        corrected_essay = corrected_essay.replace(' .', '.');

        return corrected_essay

    def correctWord(self, previous, word, next):
        if word.lower() in SpellCorrector.dictionary or not word.isalpha():
            return word

        options = SpellCorrector.suggestions[word.lower()]
        best_score = -10000;
        correction = word
        
        for option in options:
            score = self.getScore(word.lower(), previous, option, next)
            if score > best_score:
                best_score = score
                correction = option
        
        return (correction if word.islower() else correction.capitalize())

    def getScore(self, word, previous, option, next):
        unigram_prob = 0
        unigram_prob_1 = 0
        unigram_prob_2 = 0
        bigram_prob_1 = 0
        bigram_prob_2 = 0

        if len(string.split(option, ' ')) > 1:
            word_list = string.split(option, ' ');
            unigram_prob_1 = self.getUnigramProbability(word_list[0])
            unigram_prob_2 = self.getUnigramProbability(word_list[1])
            unigram_prob = min(unigram_prob_1, unigram_prob_2)
            bigram_prob_1 = self.getBigramProbability(previous, word_list[0])
            bigram_prob_2 = self.getBigramProbability(word_list[1], next)
        else:
            unigram_prob = self.getUnigramProbability(option)
            unigram_prob_1 = unigram_prob
            unigram_prob_2 = unigram_prob
            bigram_prob_1 = self.getBigramProbability(previous, option)
            bigram_prob_2 = self.getBigramProbability(option, next)

        edit_distance = self.getEditDistance(word, option, 4)

        return SpellCorrector.EDIT_WEIGHT * edit_distance + SpellCorrector.UNIGRAM_WEIGHT * math.log(unigram_prob) + SpellCorrector.BIGRAM_WEIGHT * math.log(bigram_prob_1) + SpellCorrector.BIGRAM_WEIGHT * math.log(bigram_prob_2)

    def getEditDistance(self, a, b, remaining):
        if remaining is 0:
            return 0
        if len(a) is 0 and len(b) is 0:
            return 0
        elif len(a) is 0:
            return len(b)
        elif len(b) is 0:
            return len(a)
        if a[0] is b[0]:
            return self.getEditDistance(a[1:len(a)], b[1:len(b)], remaining)
        insert_dist = self.getEditDistance(a[1:len(a)], b, remaining-1)
        delete_dist = self.getEditDistance(a, b[1:len(b)], remaining-1)
        sub_dist = self.getEditDistance(a[1:len(a)], b[1:len(b)], remaining-1)

        return 1 + min(insert_dist, delete_dist, sub_dist)

    def getBigramProbability(self, first, second):
        if first is None or second is None:
           return 1.0/SpellCorrector.bigrams[SpellCorrector.TOTAL_COUNT]
        if (first + ' ' + second) not in SpellCorrector.bigrams:
            return 1.0/SpellCorrector.bigrams[SpellCorrector.TOTAL_COUNT]
        if first not in SpellCorrector.unigrams:
            return float(SpellCorrector.bigrams[first + ' ' +  second])/(SpellCorrector.unigrams[SpellCorrector.TOTAL_COUNT]/len(SpellCorrector.unigrams)) 
        if first + ' ' + second in SpellCorrector.bigrams:
            return float(SpellCorrector.bigrams[first + ' ' +  second])/SpellCorrector.unigrams[first] 
        else:
           return 1.0/SpellCorrector.bigrams[SpellCorrector.TOTAL_COUNT]

    def getUnigramProbability(self, word):
        if word in SpellCorrector.unigrams:
            return float(SpellCorrector.unigrams[word])/SpellCorrector.unigrams[SpellCorrector.TOTAL_COUNT]
        else:
            return 1.0/SpellCorrector.unigrams[SpellCorrector.TOTAL_COUNT]

