DEBUG = True
ESSAY_SETS = range(1, 9)

GRANULAR_MODELS = True

#(deprecated)TOTAL_WORD_BIGRAMS = 50
# TOTAL_WORD_UNIGRAMS = 25
#(deprecated)TOTAL_POS_UNIGRAMS = 10
#(deprecated)TOTAL_POS_BIGRAMS = 10
LSI_TOPICS = 25 # 50
POS_LSI_TOPICS = 10 # 10

# For faster testing, set these to power_iters=2, extra_samples=100
LSI_POWER_ITERS = 2
LSI_EXTRA_SAMPLES = 100

FEATURE_CACHE = {
    'FeatureHeuristics': True,
    'FeatureSpelling': True,
    'FeatureTransitions': True,
    'FeatureLSI': True, # If set to False, I highly suggest setting genLSA to False.
    'FeaturePrompt': True,
    'FeaturePOS_LSI': True,
    'genLSA': True,
    'FeatureSim': True,
    }
