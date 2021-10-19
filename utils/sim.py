"""
Utility functions for simulating data.
"""

import numpy as np
import pandas as pd

from scipy.stats import norm
from sklearn.datasets import make_regression


def create_covariates(n_samples, seed=0, **kwargs):
    """
    Generates a covariate matrix of the given number of samples and features.

    Args:
        n_samples (int): number of samples to generate
        seed (int): seed for reproducibility
        **kwargs (dict): other keyword arguments passed to make_regression

    Returns:
        pd.DataFrame
    """
    #print(kwargs)

    X, y = make_regression(n_samples=n_samples, random_state=seed, **kwargs)

    feat_cols = ["feat_" + str(x) for x in range(X.shape[1])]

    df = pd.DataFrame(X, columns=feat_cols)

    # min max scale to [0, 1]
    # change 0 to -1 to scale to [-1, 1]
    rescale_y = 0 + ((y - np.min(y))*(1 - (0)) / (np.max(y) - np.min(y)))

    df['comply_coeff'] = rescale_y

    return df


def generate_IV_comply_indicator(n_samples, tau, seed=0,
                                 prop_nt=0.4, prop_at=0.4, prop_z=0.5,
                                 C_T=0.8, Z_T=0.8,
                                 use_covars=False,
                                 regression_dict={}):
    """
    Generates IV data with perfect compliance indicator X.

    Note from Baocchi et al. 2014 Eq 16 that "instrument strength" is equivalent to proportion of compliers.


    See this for simulating IVs:
        - https://statmodeling.stat.columbia.edu/2019/06/20/how-to-simulate-an-instrumental-variables-problem/

    Args:
        n_samples (int): the number of samples to generate
        tau (float): the treatment effect on compliers to measure
        seed (int): see for reproducibility
        prop_nt (float): the proportion of never takers
        prop_at (float): the proportion of always takers, note that prop_comply is then 1 - prop_nt - prop_at
        prop_z (float): the proportion of individuals "encouraged" by the instrument
        C_T (float): the covariance between the confounder and treatment; only applies to compliers
                     TODO is this an issue?
        Z_T (float): the covariance between the instrument and treatment
        use_covars (bool): whether or not to generate covariates that determine compliance
        regression_dict (dict): kwargs to pass to sklearn.make_regression, only used when use_covars=True

    Returns:
        feat_df (pd.DataFrame): dataframe containing all columns for IV analysis
    """
    assert (prop_nt + prop_at) < 1, "proportion of compliers needs to be > 0"

    if use_covars:
        feat_df = create_covariates(n_samples=n_samples, seed=seed, **regression_dict)

        loc, scale = norm.fit(feat_df['comply_coeff'])

        def comply_indicator(x):
            if x < norm.ppf(prop_nt, loc=loc, scale=scale):
                return 'nt'
            elif x < norm.ppf(prop_nt + prop_at, loc=loc, scale=scale):
                return 'at'
            else:
                return 'co'
    else:
        np.random.seed(seed)
        feat_df = pd.DataFrame()
        feat_df['comply_coeff'] = np.random.uniform(0, 1, size=n_samples)

        def comply_indicator(x):
            if x < prop_nt:
                return 'nt'
            elif x < prop_nt + prop_at:
                return 'at'
            else:
                return 'co'

    feat_df['comply_status'] = feat_df['comply_coeff'].apply(comply_indicator)
    X = (feat_df['comply_status'] == 'co').astype(int)

    # vars:             Z    T    C
    covar = np.array([[1.0, Z_T, 0.0], # Z
                      [Z_T, 1.0, C_T], # T
                      [0.0, C_T, 1.0]])# C

    covar += np.eye(3,3)

    # vars:  Z  T  C
    means = [0, 0, 0]

    # generate Z, T, Z
    data = np.random.multivariate_normal(mean=means, cov=covar, size=n_samples)

    # generate binary instrument
    Z = (data[:, 0] > norm.ppf(1-prop_z)).astype(int)

    # generate endogenous treatment
    T = (data[:, 1] > 0).astype(int)

    # fill in Z when the sample is a complier, 0 for nt, 1 for at
    T = np.where(feat_df['comply_status'] == 'co', Z, T)
    T = np.where(feat_df['comply_status'] == 'nt', 0, T)
    T = np.where(feat_df['comply_status'] == 'at', 1, T)

    C = data[:, 2]

    # add noncomplier bias, currently symmetric for never-takers and always-takers
    nc_bias_eff = 0.25
    B = np.where(X != 1, nc_bias_eff, 0)

    #Y = (tau + B)*T + C + np.random.normal(0, 1, n_samples)

    # optionally add heteroskedastic noise
    Y = (tau + B)*T + C + np.random.normal(0, 1, n_samples) + ((1-X) * np.random.normal(0, 1, n_samples))

    feat_df['Z'] = Z
    feat_df['X'] = X
    feat_df['T'] = T
    feat_df['C'] = C
    feat_df['Y'] = Y
    feat_df['B'] = B

    return feat_df