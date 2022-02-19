"""
Functions for compliance classification.
"""
import numpy as np

from econml.grf import CausalForest
from sklearn.model_selection import KFold

def iv_neff(scores, tz_df, threshold):
    """
    Computes the effective sample size of the given data sample.
    
    Args:
        scores (np.array): array of compliance scores
        tz_df (pd.DataFrame)
    
    """    
    tz_df['score'] = scores
    sel_df = tz_df[tz_df['score'] >= threshold].copy()
    
    if (sel_df[(sel_df['Z'] == 1)].shape[0] == 0) or (sel_df[(sel_df['Z'] == 0)].shape[0] == 0):
        #print("no samples!")
        return -np.inf
    
    comply_rate = sel_df[(sel_df['Z'] == 1)]['T'].mean() - sel_df[(sel_df['Z'] == 0)]['T'].mean()
    
    return sel_df.shape[0] * (comply_rate**2)


class ThresholdCV():
    """
    Custom CV operation to detect the best threshold.
    
    Could also be converted to an estimator to be nested with GridSearchCV, 
    down the line.
    """
    default_n_thresholds = 100
    random_state = 42
    
    def __init__(self, thresholds=None, n_splits=5, clf_class=None):
        self.thresholds = thresholds
        self.n_splits = n_splits
        
        if clf_class:
            self.clf_class = clf_class
        else:
            self.clf_class = CausalForest
        
    def get_best_threshold(self, iv_df, feat_cols):
        """
        Returns the best threshold based on KFold fitting.
        
        Args:
            iv_df (pd.DataFrame): the IV data
            feat_cols (list): the list of column name features
        """
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        # each row is the neff score for all thresholds on a particular fold
        # each column holds all the fold scores for a particular threshold
        if self.thresholds is not None:
            neff_scores = np.zeros((self.n_splits, len(self.thresholds)))
        else:
            neff_scores = np.zeros((self.n_splits, self.default_n_thresholds))
        
        for split_idx, (train_idx, test_idx) in enumerate(kfold.split(iv_df)):
            train_df = iv_df.iloc[train_idx].copy()
            test_df = iv_df.iloc[test_idx].copy()
            
            # train a single classifier for a given split
            train_X = train_df[feat_cols]
            train_Y = train_df['T']
            train_T = train_df['Z']
            
            test_X = test_df[feat_cols]
            test_Y = test_df['T']
            test_T = test_df['Z']
            
            # initialize the class constructor
            clf = self.clf_class(random_state=self.random_state,
                                 n_estimators=500,
                                 max_features=None,
                                 max_depth=3
                                 )
            clf.fit(X=train_X, y=train_Y, T=train_T)
            
            scores = clf.predict(test_X)
            tz_df = test_df[['T', 'Z']].copy()
            
            # go over all thresholds with fixed clf
            if self.thresholds is None:
                self.thresholds = np.linspace(np.min(scores), np.max(scores), 
                                              num=self.default_n_thresholds).flatten()
            
            
            for thres_idx, threshold in enumerate(self.thresholds):
                neff = iv_neff(scores, tz_df, threshold)
                neff_scores[split_idx, thres_idx] = neff
    
        mean_neff = np.mean(neff_scores, axis=0)
        best_threshold = self.thresholds[np.argmax(mean_neff)]
        
        assert mean_neff.shape[0] == len(self.thresholds)
        
        #print("Best neff: {:.3f}".format(np.max(mean_neff)))
        #print("Best threshold: {:.3f}".format(best_threshold))
        return best_threshold
                

