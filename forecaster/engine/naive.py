#!/usr/bin/env python
# coding: utf8

import numpy as np

from forecaster.engine.engine import *


class Naive(Engine):
    def predict(self):
        for i in range(len(self.validation.columns)):
            if i == 0:
                self.predictions.append(self.training[self.training.columns[-1]].values)
            else:
                self.predictions.append(self.validation[self.validation.columns[i - 1]].values)

        self.predictions = np.transpose(np.array([row.tolist() for row in self.predictions]))
