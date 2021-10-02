"""
Utility functions for power calculations.
"""

import numpy as np
from scipy.stats import norm


# old imports
# import pandas as pd
# from scipy.special import logit
# from linearmodels.iv import IV2SLS
# import seaborn as sns
# import statsmodels.api as sm
# import matplotlib.pyplot as plt


def rdd_power(effect, var, bias=0, alpha=0.05):
    """
    Implements the power function of the LATE as described in
    Cattaneo et al. 2019.

    Assumes a two-sided hypothesis test.

    Params:
        effect (float): desired LATE to be detected (against a no effect null)
        var (float): the estimated variance of the treatment effect estimator
        bias (float): estimated misspecification bias of the estimator,
                      defaults to 0
        alpha (float): significance level, defaults to 0.05
    Returns:
        power (float): the power of the specified setup
    """
    lower = norm.cdf((effect + bias) / np.sqrt(var) + norm.ppf(1 - (alpha/2)))
    upper = norm.cdf((effect + bias) / np.sqrt(var) - norm.ppf(1 - (alpha/2)))
    power = 1 - lower + upper

    return power


def iv_power(iv_df, tau, X=None, robust_se=False):
    """
    Wrapper for R IVmodel power calculation.

    Args:
        iv_df (pd.df): dataframe with Y, T, and Z columns
        tau (float): the treatment effect to calculate power against
        X (pd.Series): optionally provide covariates to control for
        robust_se (bool): whether to compute robust standard errors

    Returns:
        float: estimated power given the data and treatment effect
    """
    # Moving the import here because of weird static TLS error
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri

    # Rpy2 config: must be activated TODO better way to do this?
    pandas2ri.activate()
    ivmodel_r = importr('ivmodel')

    if X is None:
        iv_r = ivmodel_r.ivmodel(Y=iv_df['Y'],
                                 D=iv_df['T'],
                                 Z=iv_df['Z'],
                                 heteroSE=robust_se)
    else:
        iv_r = ivmodel_r.ivmodel(Y=iv_df['Y'],
                                 D=iv_df['T'],
                                 Z=iv_df['Z'],
                                 X=X,
                                 heteroSE=robust_se)

    return ivmodel_r.IVpower(iv_r, beta=tau)[0]

