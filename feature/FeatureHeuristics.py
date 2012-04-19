import abc
from FeatureBase import FeatureBase
import math
import nltk
import numpy as np
import re

class FeatureHeuristics(object):
    
    word_splitter = re.compile("\\s+")
    sentence_splitter = nltk.PunktSentenceTokenizer()
    entity_types = ["PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME", "MONEY", "PERCENT"]
    notable_punctuation = ['.', '?', '!', '-', ';']
    
    
    ### Taken from nltk_contrib
    ###
    ### Fallback syllable counter
    ###
    ### This is based on the algorithm in Greg Fast's perl module
    ### Lingua::EN::Syllable.
    ###

    specialSyllables_en = """tottered 2
    chummed 1
    peeped 1
    moustaches 2
    shamefully 3
    messieurs 2
    satiated 4
    sailmaker 4
    sheered 1
    disinterred 3
    propitiatory 6
    bepatched 2
    particularized 5
    caressed 2
    trespassed 2
    sepulchre 3
    flapped 1
    hemispheres 3
    pencilled 2
    motioned 2
    poleman 2
    slandered 2
    sombre 2
    etc 4
    sidespring 2
    mimes 1
    effaces 2
    mr 2
    mrs 2
    ms 1
    dr 2
    st 1
    sr 2
    jr 2
    truckle 2
    foamed 1
    fringed 2
    clattered 2
    capered 2
    mangroves 2
    suavely 2
    reclined 2
    brutes 1
    effaced 2
    quivered 2
    h'm 1
    veriest 3
    sententiously 4
    deafened 2
    manoeuvred 3
    unstained 2
    gaped 1
    stammered 2
    shivered 2
    discoloured 3
    gravesend 2
    60 2
    lb 1
    unexpressed 3
    greyish 2
    unostentatious 5
    """
    

    def __init__(self):
        self.features = np.array(())
        self.type = 'real'
        
        self.fallback_cache = {}

        self.fallback_subsyl = ["cial", "tia", "cius", "cious", "gui", "ion", "iou",
                               "sia$", ".ely$"]

        self.fallback_addsyl = ["ia", "riet", "dien", "iu", "io", "ii",
                                "[aeiouy]bl$", "mbl$",
                                "[aeiou]{3}",
                                "^mc", "ism$",
                                "(.)(?!\\1)([aeiouy])\\2l$",
                                "[^l]llien",
                                "^coad.", "^coag.", "^coal.", "^coax.",
                                "(.)(?!\\1)[gq]ua(.)(?!\\2)[aeiou]",
                                "dnt$"]
                                
        # Compile our regular expressions
        for i in range(len(self.fallback_subsyl)):
            self.fallback_subsyl[i] = re.compile(self.fallback_subsyl[i])
        for i in range(len(self.fallback_addsyl)):
            self.fallback_addsyl[i] = re.compile(self.fallback_addsyl[i])

        # Read our syllable override file and stash that info in the cache
        for line in FeatureHeuristics.specialSyllables_en.splitlines():
            line = line.strip()
            if line:
                toks = line.split()
                assert len(toks) == 2
                self.fallback_cache[self._normalize_word(toks[0])] = int(toks[1])
                
                
    def _normalize_word(self, word):
        return word.strip().lower()

    def countSyllables(self, word):
        word = self._normalize_word(word)

        if not word:
            return 0

        # Check for a cached syllable count
        count = self.fallback_cache.get(word, -1)
        if count > 0:
            return count

        # Remove final silent 'e'
        if word[-1] == "e":
            word = word[:-1]

        # Count vowel groups
        count = 0
        prev_was_vowel = 0
        for c in word:
            is_vowel = c in ("a", "e", "i", "o", "u", "y")
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel

        # Add & subtract syllables
        for r in self.fallback_addsyl:
            if r.search(word):
                count += 1
        for r in self.fallback_subsyl:
            if r.search(word):
                count -= 1

        # Cache the syllable count
        self.fallback_cache[word] = count

        return count

    def countComplexWords(self, words):
        complexWords = [word for word in words if self.countSyllables(word) >= 3]
        return len(complexWords)
        
    def countEntitiesOfType(self, type, line):
        """Return the number of entities of the specified type in the given line."""
        return line.count("@%s" %type.upper())

    def numFeatures(self):
        """Return the number of features for this feature vector"""
        return length(self.features)

    def featureType(self):
        """Returns string description of the feature type, such as 'real-valued', 'binary', 'enum', etc."""
        return self.type

    def getFeatureMatrix(self):
        """Returns numpy matrix of features"""
        return self.features

    def extractFeatures(self, ds, corpus):
        """Extracts features from a DataSet ds"""
        lenfeats = list()
        for line in ds.getRawText():
            curfeat = list()
            words = (FeatureHeuristics.word_splitter.split(line))
            syllables = [self.countSyllables(word) for word in words]
            sentences = FeatureHeuristics.sentence_splitter.tokenize(line)
            numChars = float(len(line))
            numWords = float(len(words))
            numComplexWords = float(self.countComplexWords(words))
            numSentences = float(len(sentences))

            num5Words = 0
            num6Words = 0
            num7Words = 0
            num8Words = 0

            for word in words:
                wordlen = len(word)
                if(wordlen > 5):
                    num5Words += 1
                    if(wordlen > 6):
                        num6Words += 1
                        if(wordlen > 7):
                            num7Words += 1
                            if(wordlen > 8):
                                num8Words += 1

            curfeat.append(numChars)
            curfeat.append(numWords)
            curfeat.append(len(set(words)))
            curfeat.append(math.pow(numWords, 0.25))
            curfeat.append(numSentences)
            curfeat.append(numChars/numWords)
            curfeat.append(numWords/numSentences)
            
            # Motivated by readability metrics like: http://en.wikipedia.org/wiki/Coleman-Liau_Index (and related)
            curfeat.append(numSentences/numWords)
            curfeat.append(sum(syllables) / numWords)
            curfeat.append(numComplexWords / numWords)
            curfeat.append(np.sqrt(numComplexWords / numSentences))
            
            curfeat.append(num5Words)
            curfeat.append(num6Words)
            curfeat.append(num7Words)
            curfeat.append(num8Words)
            #capwords = [word for word in words if word == word.capitalize()]
            #curfeat.append(len(capwords))
            curfeat.append(words.count('I'))
            curfeat.append(words.count('i'))
            
            # Types of entities in the essays
            for entity_type in self.entity_types:
                # curfeat.append(self.countEntitiesOfType(entity_type, line))    
                # curfeat.append(1 if self.countEntitiesOfType(entity_type, line) > 0 else 0)    
                pass
            curfeat.append(1 if line.count('@') > 0 else 0)
            
            for punct in self.notable_punctuation:
                curfeat.append(line.count(punct))
                
            
                
            #print curfeat

            lenfeats.append(curfeat)

        self.features = np.asarray(lenfeats)
        return

FeatureBase.register(FeatureHeuristics)

