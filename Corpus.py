# Corpus: for use in getting distributional information about text, such as unigrams/bigrams/trigrams

import nltk
from nltk.collocations import *
import DataSet
import LanguageUtils
import logging, gensim, bz2
import cPickle as pickle
from collections import Counter
import params

if params.DEBUG:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

CORPUS_CACHE = True

class Corpus:
    def __init__(self):
        self.corpus = list()
        self.pos_corpus = list()
        self.lsi = None
        self.tfidf = None
        self.pos_tfidf = None
        self.pos_lsi = None
        self.ds1 = None
        self.ds2 = None
        self.word_dicionary = None

    def getWordDictionary(self):
        """As degined for LSI (see genLSI for details)."""
        return self.word_dictionary

    def getNGrams(self):
        return self.corpus

    def getAllPOS(self):
        return self.pos_corpus

    def setCorpus(self, ds, ds2):
        self.ds1 = ds
        self.ds2 = ds2

        c = list()
        for bow in ds.getAllBoW():
            c += bow

        if ds2:
            for bow in ds2.getAllBoW():
                c += bow

        self.corpus = c

        pos_corpus = list()
        for line in ds.getAllPOS():
            pos_corpus += line

        if ds2:
            for line in ds2.getAllPOS():
                pos_corpus += line

        self.pos_corpus = pos_corpus

    def genPOS_LSA(self):
        ds1 = self.ds1
        ds2 = self.ds2

        cache_fname = 'cache/pos_lsa.%s.%s.set%d.pickle' % (
            ds1.getFilename(),
            ds2.getFilename(),
            ds1.getEssaySet())

        try:
            if not params.FEATURE_CACHE['genLSA']:
                raise Exception('Do not cache genLSA.')
            f = open(cache_fname, 'rb')
            self.pos_lsi, self.pos_tfidf, mm_corpus, dictionary = pickle.load(f)

            # split into two corpii (haha) and ds?.setGensimCorpus(mm)
            ds1.setGensimPOSCorpus(mm_corpus[0:ds1.size()])
            ds2.setGensimPOSCorpus(mm_corpus[ds1.size():(ds1.size()+ds2.size())])

            return
        except:
            pass

        documents = list()
        for tags in ds1.getAllPOS():
            documents.append(tags)
        for tags in ds2.getAllPOS():
            documents.append(tags)

        # TODO remove single appearances?

        dictionary = gensim.corpora.Dictionary(documents)
        mm_corpus = [dictionary.doc2bow(doc) for doc in documents]
        self.pos_tfidf = gensim.models.TfidfModel(mm_corpus)

        ds1.setGensimPOSCorpus(mm_corpus[0:ds1.size()])
        ds2.setGensimPOSCorpus(mm_corpus[ds1.size():(ds1.size()+ds2.size())])

        self.pos_lsi = gensim.models.LsiModel(self.pos_tfidf[mm_corpus], id2word=dictionary, num_topics=params.POS_LSI_TOPICS, power_iters=params.LSI_POWER_ITERS, extra_samples=params.LSI_EXTRA_SAMPLES)

        pickle.dump((self.pos_lsi, self.pos_tfidf, mm_corpus, dictionary), open(cache_fname, 'w'))

    def genLSA(self):
        # To cache...
        # self.lsi, self.tfidf, mm_corpus, self.word_dictionary

        ds1 = self.ds1
        ds2 = self.ds2

        cache_fname = 'cache/lsa.%s.%s.set%d.pickle' % (
            ds1.getFilename(),
            ds2.getFilename(),
            ds1.getEssaySet())

        try:
            if not params.FEATURE_CACHE['genLSA']:
                raise Exception('Do not cache genLSA.')
            f = open(cache_fname, 'rb')
            self.lsi, self.tfidf, mm_corpus, self.word_dictionary = pickle.load(f)
        
            # split into two corpii (haha) and ds?.setGensimCorpus(mm)
            ds1.setGensimCorpus(mm_corpus[0:ds1.size()])
            ds2.setGensimCorpus(mm_corpus[ds1.size():(ds1.size()+ds2.size())])

            return
        except:
            pass

        documents = list()
        bows = ds1.getAllBoW()
        #only if we want to use all in one: tags = ds1.getAllPOS()
        for i in range(0, len(bows)):
            documents.append(bows[i])
        bows = ds2.getAllBoW()
        tags = ds2.getAllPOS()
        for i in range(0, len(bows)):
            documents.append(bows[i])

        # Remove stop words
        i = 0
        size = len(documents)
        texts = list()

        # You'd be amazed at how much faster dictionaries are over lists :)
        stop_words = nltk.corpus.stopwords.words('english')
        stop_dict = dict()
        for word in stop_words:
            stop_dict[word] = 1

        # TODO move this to a different file
        stop_dict['&'] = 1

        for doc in documents:
            # TODO following line assumes BIGRAMS
            texts.append([word for word in doc if (type(word).__name__ == 'str' and word not in stop_dict) or
                                                  (type(word).__name__ == 'tuple' and word[0] not in stop_dict and word[1] not in stop_dict)])
            i += 1

        # Remove words that appear only once (TODO - maybe different min freq?)
        counts = Counter(self.getNGrams())

        i = 0
        size = len(texts)
        new_texts = list()
        for text in texts:
            new_texts.append([word for word in text if counts[word] > 1])
            i += 1
        texts = new_texts

        # create overall corpus
        dictionary = gensim.corpora.Dictionary(texts)
        self.word_dictionary = dictionary
        #dictionary.save(TODO)

        mm_corpus = [dictionary.doc2bow(text) for text in texts]
        self.tfidf = gensim.models.TfidfModel(mm_corpus)

        # split into two corpii (haha) and ds?.setGensimCorpus(mm)
        ds1.setGensimCorpus(mm_corpus[0:ds1.size()])
        ds2.setGensimCorpus(mm_corpus[ds1.size():(ds1.size()+ds2.size())])

        self.lsi = gensim.models.LsiModel(self.tfidf[mm_corpus], id2word=dictionary, num_topics=params.LSI_TOPICS,
                                          power_iters=params.LSI_POWER_ITERS, extra_samples=params.LSI_EXTRA_SAMPLES)

        f = open(cache_fname, 'w')
        pickle.dump((self.lsi, self.tfidf, mm_corpus, self.word_dictionary), f)

        return

    def getLSA(self):
        return self.lsi

    def getPOS_LSA(self):
        return self.pos_lsi

    def getTfidf(self):
        return self.tfidf

    def getPOS_Tfidf(self):
        return self.pos_tfidf

    def getTrain(self):
        return self.ds1

    def getTest(self):
        return self.ds2
