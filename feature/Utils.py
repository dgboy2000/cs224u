# Utility functions for dealing with FeatureBase-style objects

import numpy as np

def combine_features(ds, feat_list):
    final_mat = np.zeros((ds.size(), 0))

    # Each 'feat' is an implementation of 'FeatureBase'
    for feat in feat_list:
        final_mat = np.concatenate((final_mat, feat.getFeatureMatrix()), axis=1)

    return final_mat
