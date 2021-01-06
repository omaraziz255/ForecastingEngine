from Engine.Engine import *
from statsmodels.tsa.api import Holt


class HoltLinear(Engine):
    def __init__(self, preprocessed_data, window):
        super().__init__(preprocessed_data)
        self.window = window

    def predict(self):
        for row in self.training[self.training.columns[-self.window:]].values[:3]:
            fit = Holt(row).fit(smoothing_level=0.3, smoothing_slope=0.01)
            self.predictions.append(fit.forecast(self.window))

        self.predictions = np.array(self.predictions).reshape((-1, self.window))
