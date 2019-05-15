import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator


def get_pipeline():
    # Average CV score on the training set was:0.6702795787302108
    exported_pipeline = make_pipeline(
        StackingEstimator(
            estimator=GradientBoostingClassifier(learning_rate=0.001, max_depth=8, max_features=0.1, min_samples_leaf=7,
                                                 min_samples_split=8, n_estimators=100, subsample=1.0)),
        LogisticRegression(C=0.5, dual=False, penalty="l1")
    )
