import nltk

def tokenize(text):
    tokens = nltk.tokenize.WhitespaceTokenizer().tokenize(text)
    #tokens = [w.lower().strip(',.@:') for w in tokens]
    return tokens
