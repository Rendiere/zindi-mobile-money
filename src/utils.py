import os
import pandas as pd
from math import radians, cos, sin, asin, sqrt

from config import raw_data_dir


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r


def make_sub(probs):
    test_sub = pd.read_csv(os.path.join(raw_data_dir, 'sample_submission.csv'), index_col=0)
    probs_df = pd.DataFrame(probs, columns=list(test_sub))
    probs_df.index = test_sub.index.values

    return probs_df
