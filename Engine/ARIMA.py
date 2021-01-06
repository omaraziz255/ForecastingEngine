from Engine.Engine import *
from statsmodels.tsa.api import SARIMAX


class ARIMA(Engine):
    def __init__(self, preprocessed_data, window):
        super().__init__(preprocessed_data)
        self.window = window

    def predict(self):
        for row in self.training[self.training.columns[-self.window:]].values[:3]:
            fit = SARIMAX(row, seasonal_order=(0, 1, 1, 7)).fit()
            self.predictions.append(fit.forecast(self.window))
        self.predictions = np.array(self.predictions).reshape((-1, self.window))
