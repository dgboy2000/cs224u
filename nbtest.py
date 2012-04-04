from collections import Counter
from feature import FeatureUnigram
import DataSet
import nltk
import LanguageUtils
import Corpus
from feature import FeaturePrompt

ds_train = DataSet.DataSet(True)
ds_train.importData('data/c_train.utf8ignore.tsv', essay_set=1, domain_id=1)

ds_val = DataSet.DataSet(False)
ds_val.importData('data/c_val.utf8ignore.tsv', essay_set=1, domain_id=1)

corpus = Corpus.Corpus()
corpus.setCorpus(ds_train, ds_val)
#corpus.genLSA()
#corpus.genPOS_LSA()

documents = list()
bows = ds_train.getAllBoW()
for i in range(0, len(bows)):
    documents.append(bows[i])
bows = ds_val.getAllBoW()
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

for doc in documents:
    texts.append([word for word in doc if ((word not in stop_dict) and (type(word).__name__ == 'tuple'))])
    i += 1

# Remove words that appear only once (TODO - maybe different min freq?)
counts = Counter(corpus.getNGrams())

ngramset = set([word for word in corpus.getNGrams() if counts[word] > 100 and type(word).__name__ == 'tuple'])

i = 0
size = len(texts)
new_texts = list()
for text in texts:
    rel_words = [word for word in text if counts[word] > 100]
    rel_dict = dict()
    for word in ngramset:
        rel_dict[word] = False
    for word in rel_words:
        rel_dict[word] = True
    new_texts.append(rel_dict)
    i += 1
texts = new_texts

cnt = 0
grades = ds_train.getGrades()
featsets = list()
for text in texts[0:ds_train.size()]:
    featsets.append((text, grades[cnt]))
    cnt = cnt+1

classifier = nltk.NaiveBayesClassifier.train(featsets, estimator=nltk.probability.ELEProbDist)

pgrades = list()
for text in texts[ds_train.size():(ds_train.size()+ds_val.size())]:
    pgrades.append(classifier.classify(text))

print pgrades

print ds_val.getGrades()

import pdb;pdb.set_trace()

