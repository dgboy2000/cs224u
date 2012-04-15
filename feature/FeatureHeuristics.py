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

    def __init__(self):
        self.features = np.array(())
        self.type = 'real'
        
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
            sentences = FeatureHeuristics.sentence_splitter.tokenize(line)
            numChars = len(line)
            numWords = len(words)
            numSentences = len(sentences)

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
            
            
            #print curfeat

            lenfeats.append(curfeat)

        self.features = np.asarray(lenfeats)
        return

FeatureBase.register(FeatureHeuristics)

