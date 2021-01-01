from Engine.Forecast import *
from fbprophet import Prophet
import pandas as pd


class FBProphet(Forecast):
    def __init__(self, preprocessed_data, window):
        super().__init__(preprocessed_data)
        self.window = window

    def predict(self):
        ph = ["2007-12-" + str(i) for i in range(1, 31)]
        for row in self.training[self.training.columns[-self.window:]].values[:3]:
            df = pd.DataFrame(np.transpose([ph, row]))
            df.columns = ["ds", "y"]
            model = Prophet(daily_seasonality=True)
            model.fit(df)
            future = model.make_future_dataframe(periods=self.window)
            forecast = model.predict(future)["yhat"].loc[self.window:].values
            self.predictions.append(forecast)
        self.predictions = np.array(self.predictions).reshape((-1, self.window))
