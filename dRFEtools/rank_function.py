"""
This script ranks features within the feature elimination loop.
Originally developed by Tarun Katipalli.
Edits and package management by Kynon Jade Benjamin
"""

__author__ = 'Tarun Katipalli'

import numpy as np
import pandas as pd


def features_rank_fnc(features, rank, n_features_to_keep, fold, out_dir, RANK):
    """
    Args:
    features: A vector of feature names
    rank: A vector with feature ranks based on absolute value of
          feature importance
    n_features_to_keep: Number of features to keep. (Int)
    fold: Fold to analyzed. (Int)
    out_dir: Output directory for text file. Default '.'

    returns:
    Text file: Ranked features by fold tab-delimitated text file
    """
    if RANK:
        jj = n_features_to_keep + 1
        eliminated = rank[n_features_to_keep:]
        if len(eliminated) == 1:
            rank_df = pd.DataFrame({'Geneid': features[eliminated],
                                    'Fold': fold,
                                    'Rank': n_features_to_keep+1})
        elif len(eliminated) == 0:
            rank_df = pd.DataFrame({'Geneid': features[rank],
                                    'Fold': fold,
                                    'Rank': 1})
        else:
            rank_df = pd.DataFrame({'Geneid': features[eliminated],
                                    'Fold': fold,
                                    'Rank': np.arange(jj, jj+len(eliminated))})
        rank_df.sort_values('Rank', ascending=False)\
               .to_csv(out_dir+'/rank_features.txt', sep='\t', mode='a',
                       index=False, header=False)
