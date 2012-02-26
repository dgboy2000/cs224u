#http://blog.quibb.org/2009/04/spell-checking-in-python/

import os
import pickle
import popen2
 
class SpellChecker:

    spelling_cache_filename = 'spelling_cache.pkl'
    if os.path.exists(spelling_cache_filename):
        spelling_cache = pickle.load(spelling_cache_filename)
    else:
        spelling_cache = {}

    def __init__(self):
        self._f = popen2.Popen3("hunspell")
        self._f.fromchild.readline() #skip the credit line

    def __del__(self):
        """On destruction, pickle the class's cache to disk."""
        pickle.dump(SpellChecker.spelling_cache)

    def extractSpellingSuggestions(self, word):
        if word in SpellChecker.spelling_cache:
            return SpellChecker.spelling_cache[word]

        self._f.tochild.write(word + '\n')
        self._f.tochild.flush()
        s = self._f.fromchild.readline().strip().lower()
        self._f.fromchild.readline() #skip the blank line
       
        #if len(s) == 0:
        #    raise Exception("Got empty string for word %s" %(word))

        if s == "*" or len(s) == 0:
            SpellChecker.spelling_cache[word] = None
        elif s[0] == '#':
            SpellChecker.spelling_cache[word] = []
        elif s[0] == '+':
            SpellChecker.spelling_cache[word] = None
            #raise Exception("Spelling suggestion for %s in unknown format: %s" %(word, s))
        else:
            SpellChecker.spelling_cache[word] = s.split(':')[1].strip().split(', ')
        return SpellChecker.spelling_cache[word]
    
