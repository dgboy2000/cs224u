DEBUG = True
ESSAY_SETS = range(1, 9)

# Learning parameters
LEARNER_CLASS = 'LinearRegression' # 'LinearRegression' | 'MatlabExample'
REGULARIZATION = 0.01 # The value of the regularization parameter in the learning method, or None if no regularization
GRANULAR_MODELS = False
FEATURE_SELECTION = None # None | 'inclusive' | 'exclusive'
DUMP = False # True | False  Dump features and grades to file and quit

#(deprecated)TOTAL_WORD_BIGRAMS = 50
# TOTAL_WORD_UNIGRAMS = 25
#(deprecated)TOTAL_POS_UNIGRAMS = 10
#(deprecated)TOTAL_POS_BIGRAMS = 10
LSI_TOPICS = 25 # 50
POS_LSI_TOPICS = 10 # 10

# For faster testing, set these to power_iters=2, extra_samples=100
LSI_POWER_ITERS = 2
LSI_EXTRA_SAMPLES = 100

# FeatureNN - number of nearest neighbors counted with LSI similarity
NUM_NN = 25

FEATURE_CACHE = {
    'FeatureHeuristics': True,
    'FeatureSpelling': True,
    'FeatureTransitions': True,
    'FeatureLSI': True, # If set to False, I highly suggest setting genLSA to False.
    'FeaturePrompt': True,
    'FeaturePOS_LSI': True,
    'genLSA': True,
    'FeatureSim': True,
    'FeatureNN': True,
    }
