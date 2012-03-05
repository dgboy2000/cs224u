import os
import pickle

class CreateTransitions:

    transitions_txt = 'coherence/transitions.txt'
    transitions_fn = 'cache/transitions.pkl'

    def __init__(self):
        if not os.path.exists(CreateTransitions.transitions_fn):
            self.readAndPickle()

    def getTransitions(self):
        return pickle.load(open(CreateTransitions.transitions_fn, 'r'))

    def readAndPickle(self):
        f = open(CreateTransitions.transitions_txt, 'r')
        transitions = set()

        for line in f:
            word = line.split('\n')[0]
            print word
            transitions.add(word)

        pickle.dump(transitions, open(CreateTransitions.transitions_fn, 'w'))



