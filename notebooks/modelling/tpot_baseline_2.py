import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:0.6829692505801781
exported_pipeline = make_pipeline(
    MinMaxScaler(),
    StackingEstimator(estimator=BernoulliNB(alpha=1.0, fit_prior=True)),
    MaxAbsScaler(),
    SelectPercentile(score_func=f_classif, percentile=92),
    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.001, max_depth=4, max_features=0.8, min_samples_leaf=8, min_samples_split=4, n_estimators=100, subsample=0.9000000000000001)),
    MaxAbsScaler(),
    LogisticRegression(C=0.1, dual=False, penalty="l2")
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
