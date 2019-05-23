import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from pysal.lib.cg import KDTree, RADIUS_EARTH_KM

from config import location_data_dir


class MMAgentsInVicinity(BaseEstimator, TransformerMixin):

    def __init__(self, radius=5, mm_coords=None):
        if not mm_coords:
            mm_df = pd.read_csv(os.path.join(location_data_dir, 'mobilemoney_agents_for_upload_win.csv'))
            mm_coords = mm_df[['latitude', 'longitude']]

        self.tree = self.create_tree(mm_coords)
        self.radius = radius

    def create_tree(self, coords):
        if type(coords) != np.array:
            coords = np.array(coords)

        return KDTree(coords, distance_metric='ARC', radius=RADIUS_EARTH_KM)

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None):

        assert X.shape[1] == 2, 'shape of dataset passed is wrong'

        apply_fun = lambda coords: len(self.tree.query_ball_point(coords, r=self.radius))
        agents_in_radius = map(apply_fun, X)

        return pd.DataFrame(agents_in_radius)


class ColumnExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, ):
        assert self.columns is not None, 'ColumnExtractor initialized without list of columns'
        return X[self.columns]
