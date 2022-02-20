"""
Script to fit causal estimator for compliance in diabetes FRDD.
"""

import argparse
import os
import numpy as np
import pandas as pd
import sys

from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegressionCV
from econml.dml import CausalForestDML
from econml.grf import CausalForest
from econml.sklearn_extensions.model_selection import WeightedStratifiedKFold

PROJ_PATH = "../"

sys.path.append(PROJ_PATH)

from utils.clf import ThresholdCV

def split_data(df, n_splits=2, random_state=42):
    """
    Splits the data randomly, returning dataframes.

    Args:
        df (pd.DataFrame)
        n_splits (int)
        random_state (int)

    Returns:
        list of split dataframes
    """
    kfold = KFold(n_splits=n_splits,
            shuffle=True, random_state=random_state)

    splits = []

    for _, test_indices in kfold.split(df):
        split_df = df.iloc[test_indices].copy()
        splits.append(split_df)
    
    return splits

def create_features(df):
    """
    Creates features for compliance estimation.

    Returns:
        dict: X, Y, T
    """
    cont_cols = [
                 'age',
                 'days_from_2001',
                 ]
    feat_cols = df.columns[df.columns.str.startswith("d_race") | 
                           df.columns.str.startswith("d_networth") | 
                           df.columns.str.startswith("d_household") | 
                           df.columns.str.startswith("d_education_level_code") | 
                           df.columns.str.startswith("d_home_ownership") | 
                           df.columns.str.startswith("d_fed_poverty") | 
                           df.columns.str.startswith("gdr")|
                           df.columns.str.startswith("bus")|
                           df.columns.isin(cont_cols)
                           ]
    print(feat_cols)
    X = df[feat_cols].copy()
    Y = df['T'].copy()
    T = df['Z'].copy()

    # standardize age
    for col in cont_cols:
        X[col] = (X[col] - X[col].mean()) / X[col].std()

    return dict(
        X=X,
        Y=Y,
        T=T
    )


def train_excl_model(X, Y, T, model=None, random_state=42, double_ml=False):
    """
    Trains an exclusion model with the given covariates (X), outcomes (Y), and treatments (T).

    Args:
        model (econml.model): Optionally provide a model, defaults to causal forest.
        double_ml (bool): whether to use the double ML estimator
    """
    if double_ml:
        treat_model = LogisticRegressionCV(cv=WeightedStratifiedKFold(random_state=random_state),
                                 random_state=random_state,
                                 max_iter=10000,
                                 n_jobs=16,
                                 Cs=np.logspace(-6,6,num=100),
                                 )
    
        model = CausalForestDML(discrete_treatment=True,
                        cv=10,
                        n_jobs=-1,
                        model_y=clone(treat_model),
                        model_t=clone(treat_model))
        model.fit(Y=Y, T=T, X=X)
    else:
        model = CausalForest(random_state=42,
                             n_estimators=500,
                             max_features=None,
                             max_depth=3
                             )
        model.fit(y=Y, T=T, X=X)

    return model


def main():
    """
    Main function, TODO extend functionality
    """
    parser = argparse.ArgumentParser(description="script for training causal forests on harrower")
    parser.add_argument("out_path", type=str, help="the subdirectory in ../results/ to dump results to")
    parser.add_argument("--test", action='store_true', help="whether to do a test run with downsampled data")
    parser.add_argument("--dml", action='store_true', help="whether to use a double ML estimator")

    args = parser.parse_args()

    # load data
    comp_est_df = pd.read_pickle(os.path.join(PROJ_PATH, "results", args.out_path, "comp_est.df"))
    
    # only include data inside the bandwidth
    comp_est_df = comp_est_df[comp_est_df['in_bw']]
    print(comp_est_df.shape[0])

    # temporary downsample for testing
    if args.test:
        comp_est_df = comp_est_df.sample(100)

    splits = split_data(comp_est_df)

    # two splits by default
    s1_df = splits[0].copy()
    s2_df = splits[1].copy()

    s1_feat_dict = create_features(s1_df)
    s2_feat_dict = create_features(s2_df)

    s1_trained_model = train_excl_model(**s1_feat_dict, double_ml=args.dml)
    s2_trained_model = train_excl_model(**s2_feat_dict, double_ml=args.dml)

    if args.dml:
        s1_df['train_pred_comply'] = s1_trained_model.effect(s1_feat_dict['X'])
        s1_df['test_pred_comply'] = s2_trained_model.effect(s1_feat_dict['X'])

        s2_df['train_pred_comply'] = s2_trained_model.effect(s2_feat_dict['X'])
        s2_df['test_pred_comply'] = s1_trained_model.effect(s2_feat_dict['X'])

        print("s1 test score: {:.3f}".format(s2_trained_model.score(**s1_feat_dict)))
        print("s2 test score: {:.3f}".format(s1_trained_model.score(**s2_feat_dict)))
    else:
        s1_df['train_pred_comply'] = s1_trained_model.predict(s1_feat_dict['X'])
        s1_df['test_pred_comply'] = s2_trained_model.predict(s1_feat_dict['X'])

        s2_df['train_pred_comply'] = s2_trained_model.predict(s2_feat_dict['X'])
        s2_df['test_pred_comply'] = s1_trained_model.predict(s2_feat_dict['X'])

        s1_feats = pd.concat([s1_feat_dict['X'], s1_feat_dict['T'], s1_feat_dict['Y']], axis=1)
        s2_feats = pd.concat([s2_feat_dict['X'], s2_feat_dict['T'], s2_feat_dict['Y']], axis=1)
        s1_threshold = ThresholdCV().get_best_threshold(s2_feats, feat_cols=s2_feat_dict['X'].columns)
        s2_threshold = ThresholdCV().get_best_threshold(s1_feats, feat_cols=s1_feat_dict['X'].columns)
    
        s1_df['threshold'] = s1_threshold
        s2_df['threshold'] = s2_threshold

        s1_df['include'] = s1_df['test_pred_comply'] > s1_threshold
        s2_df['include'] = s2_df['test_pred_comply'] > s2_threshold


    s1_df[['patid', 'train_pred_comply', 'test_pred_comply', 'include', 'threshold']].to_pickle(os.path.join(PROJ_PATH, "results", args.out_path, "s1_pred_comply.df"))
    s2_df[['patid', 'train_pred_comply', 'test_pred_comply', 'include', 'threshold']].to_pickle(os.path.join(PROJ_PATH, "results", args.out_path, "s2_pred_comply.df"))

    #print(s1_df[['patid', 'train_pred_comply', 'test_pred_comply']].head())
    #print(s2_df[['patid', 'train_pred_comply', 'test_pred_comply']].head())

if __name__ == '__main__':
    main()