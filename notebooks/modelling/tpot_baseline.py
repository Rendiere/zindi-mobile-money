import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MaxAbsScaler
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:-0.7326581163228116
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=DecisionTreeClassifier(criterion="entropy", max_depth=2, min_samples_leaf=17, min_samples_split=4)),
    StackingEstimator(estimator=DecisionTreeClassifier(criterion="entropy", max_depth=2, min_samples_leaf=19, min_samples_split=17)),
    StackingEstimator(estimator=XGBClassifier(learning_rate=0.01, max_depth=2, min_child_weight=12, n_estimators=100, nthread=1, subsample=0.15000000000000002)),
    StackingEstimator(estimator=GaussianNB()),
    StackingEstimator(estimator=DecisionTreeClassifier(criterion="entropy", max_depth=2, min_samples_leaf=19, min_samples_split=17)),
    MaxAbsScaler(),
    VarianceThreshold(threshold=0.005),
    LogisticRegression(C=1.0, dual=False, penalty="l1")
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
