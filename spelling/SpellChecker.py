#http://blog.quibb.org/2009/04/spell-checking-in-python/

import popen2

class SpellChecker:

    def __init__(self):
        self._f = popen2.Popen3("hunspell")
        self._f.fromchild.readline() #skip the credit line

    def extractSpellingSuggestions(self, word):
        print word
        self._f.tochild.write(word + '\n')
        self._f.tochild.flush()
        s = self._f.fromchild.readline().strip().lower()
        self._f.fromchild.readline() #skip the blank line
       
        if s == "*" or len(s) == 0:
            return None
        elif s[0] == '#':
            return []
        elif s[0] == '+':
            return None
            #raise Exception("Spelling suggestion for %s in unknown format: %s" %(word, s))
        else:
            return s.split(':')[1].strip().split(', ')
        
