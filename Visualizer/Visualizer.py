from Preprocess.Preprocessor import Preprocessor
from Postprocess.Postprocessor import Postprocessor
from Data.Data import Data
from Engine.Naive import *
from Engine.MovingAverage import *
from Engine.HoltLinear import *
from Engine.ExponentialSmooth import *
from Engine.FBProphet import *
from Engine.ARIMA import *
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots


class Visualizer:
    def __init__(self):
        pass

    def display_sales(self, x, y=None):
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=np.arange(len(x)), y=x, showlegend=False, mode='lines+markers', name='Sample',
                                 marker=dict(color="lightskyblue")), row=1, col=1)
        if y is not None:
            fig.add_trace(go.Scatter(x=np.arange(len(x)), y=y, showlegend=False, mode='lines',
                                     name='Pre-processed Sample', marker=dict(color="navy")), row=1, col=1)

        fig.update_layout(height=1200, width=800, title_text=f"Sample Sales")
        fig.show()

    def roll_avg(self, sales, prices, time, per_store=False, mean=False, store=None):
        d_cols = [c for c in sales.columns if 'd_' in c]
        past_sales = sales.set_index('id')[d_cols].T.merge(time.set_index('d')['date'], left_index=True,
                                                           right_index=True, validate='1:1').set_index('date')
        stores = prices['store_id'].unique()
        means = []
        fig = go.Figure()
        if store is None:
            store = ""
        labels = []
        for s in stores:
            if store.lower() in s or store.upper() in s:
                items = [c for c in past_sales.columns if s in c]
                data = past_sales[items].sum(axis=1).rolling(90).mean()
                means.append(np.mean(past_sales[items].sum(axis=1)))
                labels.append(s)
                if per_store:
                    fig.add_trace(go.Box(x=[s] * len(data), y=data, name=s))
                else:
                    fig.add_trace(go.Scatter(x=np.arange(len(data)), y=data, name=s))
        if per_store:
            fig.update_layout(yaxis_title="Sales", xaxis_title="Stores",
                              title="Rolling Average Sales vs. Stores")
        else:
            fig.update_layout(yaxis_title="Sales", xaxis_title="Time",
                              title="Rolling Average Sales vs. Time (per store)")
        fig.show()
        if mean:
            fig = go.Figure(data=[go.Bar(name='', x=labels, y=means)])

            fig.update_layout(title="Mean sales vs. Store name", yaxis=dict(title="Mean sales"),
                              xaxis=dict(title="Store name"))
            fig.update_layout(barmode='group')
            fig.show()

    def train_val_view(self, forecast, sample, prediction=False):
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=np.arange(70), mode='lines', y=forecast.training.loc[sample].values,
                                 marker=dict(color="dodgerblue"), showlegend=False, name="Training"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=np.arange(70, 100), y=forecast.validation.loc[sample].values, mode='lines',
                                 marker=dict(color="darkorange"), showlegend=False, name="Validation"),
                      row=1, col=1)
        if prediction:
            pred = forecast.predictions[sample]
            fig.add_trace(go.Scatter(x=np.arange(70, 100), y=pred, mode='lines',
                                     marker=dict(color="seagreen"), showlegend=False, name="Prediction"),
                          row=1, col=1)
        fig.update_layout(height=1200, width=800, title_text="Train (blue) vs. Validation (orange) sales")
        fig.show()

    def visualize_loss(self, losses, names):
        fig = go.Figure(data=[go.Bar(name='', x=names, y=losses)])

        fig.update_layout(title="Losses vs. Method names", yaxis=dict(title="Losses"),
                          xaxis=dict(title="Method names"))
        fig.update_layout(barmode='group')
        fig.show()




#Test Code

d = Data("../Data/")
p = Preprocessor(d)
v = Visualizer()
x = p.load_sales(65, bgn=350, end=450)
y = p.average_smoothing(x)
# v.display_sales(x, y)

x = p.data.sales_data
y = p.data.selling_prices
z = p.data.calendar
# v.roll_avg(x, y, z, per_store=False, mean=True, store="wi")
n = Naive(p.data)
mv = MovingAverage(p.data,30)
h = HoltLinear(p.data,30)
e = ExponentialSmooth(p.data,30)
a = ARIMA(p.data,30)
ph = FBProphet(p.data, 30)

methods = [n, mv, h, e, a, ph]
for m in methods:
    m.predict()

pp = Postprocessor()
losses = []
for m in methods:
    losses.append(pp.compute_loss(m.predictions, m.validation))

names = ["Naive approach", "Moving average", "Holt linear", "Exponential smoothing", "ARIMA", "Prophet"]
v.visualize_loss(losses, names)

# v.train_val_view(ph, 0, prediction=True)
