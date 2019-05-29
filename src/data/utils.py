import os
import pandas as pd
from glob import glob

from config import data_dir, raw_data_dir, features_data_dir


def extract_targets(train_df):
    X = train_df.drop(['mobile_money', 'savings', 'borrowing', 'insurance', 'mobile_money_classification'], axis=1)
    y = train_df['mobile_money_classification']
    return X, y


def load_data():
    """
    Load training and testing sets combined with all features
    generated through feature engineering

    :return:
    """

    train_df = pd.read_csv(os.path.join(raw_data_dir, 'training.csv'), index_col=0)
    test_df = pd.read_csv(os.path.join(raw_data_dir, 'test.csv'), index_col=0)

    train_features = glob(os.path.join(features_data_dir, 'train/*.csv'))
    test_features = glob(os.path.join(features_data_dir, 'test/*.csv'))

    train_features_df = pd.concat([pd.read_csv(f, index_col=0).fillna(-1) for f in train_features], axis=1)
    test_features_df = pd.concat([pd.read_csv(f, index_col=0).fillna(-1) for f in test_features], axis=1)

    train_data = pd.concat([train_df, train_features_df], axis=1)
    test_data = pd.concat([test_df, test_features_df], axis=1)

    return train_data, test_data
