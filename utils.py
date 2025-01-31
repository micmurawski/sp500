import numpy as np
import pandas as pd

from sklearn.model_selection._split import _BaseKFold
from sklearn.base import BaseEstimator, TransformerMixin
import re
from collections import defaultdict

import numpy as np

import numpy as np
from sklearn.utils import resample

class BalancedQuarterTimeSeriesSplit:
    def __init__(self, n_splits=5, test_size=4, min_train_size=8, random_state=42, turn_off_balancing=False):
        self.n_splits = n_splits
        self.test_size = test_size
        self.min_train_size = min_train_size
        self.random_state = random_state
        self.turn_off_balancing = turn_off_balancing
        
    def _balance_set(self, X, y):
        """Balance a single set (train or test) by equal sampling"""
        # Get indices for each class
        idx_0 = np.where(y == 0)[0]
        idx_1 = np.where(y == 1)[0]
        
        # Determine size for balanced set (minimum of both classes)
        n_samples = min(len(idx_0), len(idx_1))
        
        # Sample equally from both classes
        idx_0_sampled = resample(idx_0, n_samples=n_samples, random_state=self.random_state)
        idx_1_sampled = resample(idx_1, n_samples=n_samples, random_state=self.random_state)
        
        # Combine indices
        balanced_idx = np.concatenate([idx_0_sampled, idx_1_sampled])
        np.random.shuffle(balanced_idx)
        
        return balanced_idx
        
    def split(self, X, y=None, groups=None):
        quarters = sorted(X.index.get_level_values('quarter').unique())
        n_quarters = len(quarters)
        
        if n_quarters < self.min_train_size + self.test_size:
            raise ValueError(
                f"Too few quarters ({n_quarters}) for min_train_size "
                f"({self.min_train_size}) + test_size ({self.test_size})"
            )
            
        max_splits = n_quarters - self.min_train_size - self.test_size + 1
        n_splits = min(self.n_splits, max_splits)
        
        if n_splits > 1:
            step = (n_quarters - self.min_train_size - self.test_size) // (n_splits - 1)
        else:
            step = 1
            
        for i in range(n_splits):
            test_end = n_quarters - (n_splits - 1 - i) * step
            test_start = test_end - self.test_size
            train_end = test_start
            train_start = 0
            
            train_quarters = quarters[train_start:train_end]
            test_quarters = quarters[test_start:test_end]
            
            # Get masks for initial split
            train_mask = X.index.get_level_values('quarter').isin(train_quarters)
            test_mask = X.index.get_level_values('quarter').isin(test_quarters)
            
            # Get initial train/test split
            X_train_idx = np.where(train_mask)[0]
            X_test_idx = np.where(test_mask)[0]
            
            # Balance training set
            if not self.turn_off_balancing:
                balanced_train_idx = self._balance_set(
                    X.iloc[X_train_idx], 
                    y.iloc[X_train_idx]
                )

                # Balance test set
                balanced_test_idx = self._balance_set(
                    X.iloc[X_test_idx], 
                    y.iloc[X_test_idx]
                )
            else:
                balanced_train_idx = X_train_idx
                balanced_test_idx = X_test_idx
            
            # Convert local indices back to global indices
            final_train_idx = X_train_idx[balanced_train_idx]
            final_test_idx = X_test_idx[balanced_test_idx]
            
            yield final_train_idx, final_test_idx
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits



class TemporalBalancedSplitter(BaseEstimator, TransformerMixin):    
    def __init__(self, test_size=0.2, random_state=None, min_train_periods=4):
        self.test_size = test_size
        self.random_state = random_state
        self.min_train_periods = min_train_periods
        
    def fit(self, X, y=None):
        return self
    
    def split(self, X, y):
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must have a MultiIndex with (quarter, ticker)")
            
        if isinstance(y, pd.Series):
            y = y.values
            
        quarters = sorted(X.index.get_level_values('quarter').unique())
        n_quarters = len(quarters)
        
        if n_quarters < self.min_train_periods + 1:
            raise ValueError(f"Not enough quarters for splitting. Minimum required: {self.min_train_periods + 1}")
            
        split_quarter = quarters[int(n_quarters * (1 - self.test_size))]
        
        train_idx_0 = []  # Class 0 training indices
        train_idx_1 = []  # Class 1 training indices
        test_idx_0 = []   # Class 0 test indices
        test_idx_1 = []   # Class 1 test indices
        
        for i, idx in enumerate(X.index):
            quarter = idx[0]  # Get quarter from MultiIndex
            if quarter < split_quarter:
                if y[i] == 0:
                    train_idx_0.append(i)
                else:
                    train_idx_1.append(i)
            else:
                if y[i] == 0:
                    test_idx_0.append(i)
                else:
                    test_idx_1.append(i)
        
        # Balance classes within each split
        np.random.seed(self.random_state)
        
        # Balance training set
        min_train_samples = min(len(train_idx_0), len(train_idx_1))
        train_idx_0 = np.random.choice(train_idx_0, min_train_samples, replace=False)
        train_idx_1 = np.random.choice(train_idx_1, min_train_samples, replace=False)
        
        # Balance test set
        min_test_samples = min(len(test_idx_0), len(test_idx_1))
        test_idx_0 = np.random.choice(test_idx_0, min_test_samples, replace=False)
        test_idx_1 = np.random.choice(test_idx_1, min_test_samples, replace=False)
        
        # Combine and sort indices
        self.train_idx_ = np.sort(np.concatenate([train_idx_0, train_idx_1]))
        self.test_idx_ = np.sort(np.concatenate([test_idx_0, test_idx_1]))
        
        return self.train_idx_, self.test_idx_
    
    def split_data(self, X, y):
        train_idx, test_idx = self.split(X, y) 
        if isinstance(y, pd.Series):
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
        else:
            y_train = y[train_idx]
            y_test = y[test_idx]
            
        return X.iloc[train_idx], X.iloc[test_idx], y_train, y_test
    

def select_features_correlation(
        correlation_matrix: pd.DataFrame,
        target_column: str,
        threshold: float=0.5,
        collinearity_threshold: float = 0.7,
        abs_values: bool = True,
        gt:bool = True
):
    if abs_values:
        target_correlations = abs(correlation_matrix[target_column])
    else:
        target_correlations = correlation_matrix[target_column]
    
    if gt:
        potential_features = target_correlations[target_correlations > threshold].index.tolist()
    else: # lt - less than
        potential_features = target_correlations[target_correlations < threshold].index.tolist()
    if target_column in potential_features:
        potential_features.remove(target_column)  # Remove target variable from features
    
    selected_features = []
    for feature in potential_features:
        
        add_feature = True
        
        for selected in selected_features:
            if abs(correlation_matrix.loc[feature, selected]) > collinearity_threshold:
                add_feature = False
                break
        
        if add_feature:
            selected_features.append(feature)
    
    return selected_features



def group_similar_features(features_set:list[str]) -> dict[str, list[str]]:
    grouped = defaultdict(list)

    for feat_name in features_set:
        if re.search(r'_(\d+)_', feat_name):
            group_name = re.sub(r'_(\d+)_', "_x_", feat_name)
            grouped[group_name].append(feat_name)
        else:
            grouped[feat_name] = [feat_name]
    return grouped

def pick_n_largest(correlation_matrix: pd.DataFrame, grouped: dict[str, list[str]], target_column: str, n:int) -> list[str]:
    target_correlations = correlation_matrix[target_column]
    selected = []
    for group_name in grouped:
        if len(grouped[group_name]) == 1:
            selected.append(grouped[group_name][0])
        else:
            features = grouped[group_name]
            selected += list(target_correlations[features].nlargest(n).index)
    return selected

def pick_n_smallest(correlation_matrix: pd.DataFrame, grouped: dict[str, list[str]], target_column: str, n:int) -> list[str]:
    target_correlations = correlation_matrix[target_column]
    selected = []
    for group_name in grouped:
        if len(grouped[group_name]) == 1:
            selected.append(grouped[group_name][0])
        else:
            features = grouped[group_name]
            selected += list(target_correlations[features].nsmallest(n).index)
    return selected


def pick_n_largest_abs(correlation_matrix: pd.DataFrame, grouped: dict[str, list[str]], target_column: str, n:int) -> list[str]:
    target_correlations = abs(correlation_matrix[target_column])
    selected = []
    for group_name in grouped:
        if len(grouped[group_name]) == 1:
            selected.append(grouped[group_name][0])
        else:
            features = grouped[group_name]
            selected += list(target_correlations[features].nlargest(n).index)
    return selected
