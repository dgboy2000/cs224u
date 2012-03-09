# Utility functions for dealing with FeatureBase-style objects

import numpy as np

def combine_features(ds, feat_list):
    final_mat = np.zeros((ds.size(), 0))

    # Each 'feat' is an implementation of 'FeatureBase'
    for feat_mat in feat_list:
        try:
            final_mat = np.concatenate((final_mat, feat_mat), axis=1)
        except:
            import pdb;pdb.set_trace()

    return final_mat
