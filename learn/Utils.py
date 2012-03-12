# Utility functions for learning

import numpy as np
import os
import pickle

feature_cache_path = 'cache'

def save_features(features, grades, filename='features.pickle'):
    num_essays, num_features = features.shape


