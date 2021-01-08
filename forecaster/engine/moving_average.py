from engine.forecast import *


class MovingAverage(Forecast):
    def __init__(self, preprocessed_data, window):
        super().__init__(preprocessed_data)
        self.window = window

    def predict(self):
        for i in range(len(self.validation.columns)):
            if i == 0:
                self.predictions.append(np.mean(self.training[self.training.columns[-self.window:]].values, axis=1))
            if self.window+1 > i > 0:
                self.predictions.append(0.5 * (np.mean(self.training[self.training.columns[-self.window + i:]].values,
                                                       axis=1) + np.mean(self.predictions[:i], axis=0)))
            if i > self.window+1:
                self.predictions.append(np.mean([self.predictions[:i]], axis=1))

        self.predictions = np.transpose(np.array([row.tolist() for row in self.predictions]))
