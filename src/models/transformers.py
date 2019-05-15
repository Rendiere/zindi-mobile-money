from pandas import DataFrame
from sklearn.base import TransformerMixin


class ModelTransformer(TransformerMixin):
    """
    Wrapper to use a model as a feature transformer
    Taken from http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html

    The pipeline treats these objects like any of the built-in
    transformers and fits them during the training phase,
    and transforms the data using each one when predicting.
    """

    def __init__(self, model):
        self.model = model

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        return DataFrame(self.model.predict(X))
