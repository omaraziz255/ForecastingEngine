#!/usr/bin/env python
# coding: utf8

import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing

from forecaster.engine.engine import *


class ExponentialSmooth(Engine):
    def __init__(self, preprocessed_data, window):
        super().__init__(preprocessed_data)
        self.window = window

    def predict(self):
        for row in self.training[self.training.columns[-self.window:]].values[:3]:
            fit = ExponentialSmoothing(row, seasonal_periods=3).fit()
            self.predictions.append(fit.forecast(self.window))
        self.predictions = np.array(self.predictions).reshape((-1, self.window))
