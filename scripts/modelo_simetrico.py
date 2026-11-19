"""Probabilidades de uma série independentes da ordem em que as equipes são passadas."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class ModeloSimetrico(ClassifierMixin, BaseEstimator):
    def __init__(self, modelo):
        self.modelo = modelo

    @property
    def classes_(self):
        return self.modelo.classes_

    def predict_proba(self, x):
        inverted = x.copy()
        differences = [column for column in x.columns if column.startswith('dif_')]
        inverted[differences] = -inverted[differences]
        positive = list(self.classes_).index(1)
        direct = self.modelo.predict_proba(x)[:, positive]
        reverse = self.modelo.predict_proba(inverted)[:, positive]
        probability = (direct + 1.0 - reverse) / 2.0
        result = np.empty((len(x), len(self.classes_)))
        result[:, positive] = probability
        result[:, 1-positive] = 1.0-probability
        return result

    def predict(self, x):
        return self.classes_[np.argmax(self.predict_proba(x), axis=1)]
