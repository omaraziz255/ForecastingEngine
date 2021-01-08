import pandas as pd


class Data:
    def __init__(self, input_dir):
        if input_dir is None:
            input_dir = '/'
        self.calendar, self.selling_prices, self.sales_data = None, None, None
        self.load_data(input_dir)

    def load_data(self,input_dir):
        self.calendar = pd.read_csv(f'{input_dir}calendar.csv')
        self.selling_prices = pd.read_csv(f'{input_dir}sell_prices.csv')
        self.sales_data = pd.read_csv(f'{input_dir}sales_train_validation.csv')
