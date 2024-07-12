"""
This script ranks features within the feature elimination loop.
Originally developed by Tarun Katipalli.
Edits and package management by Kynon Jade Benjamin
"""

__author__ = 'Tarun Katipalli'

import numpy as np
import pandas as pd
from os.path import join, exists

__all__ = ["features_rank_fnc"]

def features_rank_fnc(features, rank, n_features_to_keep, fold, out_dir, RANK):
    """
    Ranks features and writes the results to a file
    Args:
        features: A vector of feature names
        rank: A vector with feature ranks based on absolute value of
              feature importance
        n_features_to_keep (int): Number of features to keep.
        fold (int): Current fold being analyzed.
        out_dir (str): Output directory for text file. Default is current
                       directory.
        RANK (bool): Whether to perform ranking and write results.

    Returns:
        None

    Writes:
       Text file: Ranked features by fold tab-delimitated text file
    """
    if not RANK:
        return

    if not isinstance(n_features_to_keep, int) or n_features_to_keep < 0:
        raise ValueError("n_features_to_keep must be a non-negative integer")

    if not isinstance(fold, int) or fold < 0:
        raise ValueError("fold must be a non-negative integer")

    if len(features) != len(rank):
        raise ValueError("Length of features and rank must be the same")

    features = np.array(features)
    rank = np.array(rank)
    eliminated = rank[n_features_to_keep:]

    if len(eliminated) == 0:
        rank_df = pd.DataFrame({
            'Geneid': features[rank],
            'Fold': fold,
            'Rank': 1
        })
    else:
        rank_df = pd.DataFrame({
            'Geneid': features[eliminated],
            'Fold': fold,
            'Rank': np.arange(n_features_to_keep + 1,
                              n_features_to_keep + 1 + len(eliminated))
        })
    output_file = join(out_dir, "rank_features.txt")
    rank_df.sort_values('Rank', ascending=False).to_csv(
        output_file,
        sep='\t',
        mode='a',
        index=False,
        header=not exists(output_file)
    )
