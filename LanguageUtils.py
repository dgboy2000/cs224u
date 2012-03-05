import nltk

def tokenize(text):
    tokens = nltk.tokenize.WhitespaceTokenizer().tokenize(text)
    tokens = [w.strip(',.@:"!?/()*').lower() for w in tokens]
    return tokens

def punkt_tokenize(text):
    word_tokenizer = nltk.tokenize.WordPunctTokenizer()
    return word_tokenizer.tokenize(text)
