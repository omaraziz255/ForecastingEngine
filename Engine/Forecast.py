import abc


class Forecast:
    __metaclass__ = abc.ABCMeta

    def __init__(self, preprocessed_data):
        d_cols = [c for c in preprocessed_data.sales_data.columns if 'd_' in c]
        self.training = preprocessed_data.sales_data[d_cols[-100:-30]]
        self.validation = preprocessed_data.sales_data[d_cols[-30:]]
        self.predictions = []

    @abc.abstractmethod
    def predict(self):
        return
