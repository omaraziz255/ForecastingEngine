from Preprocess.Preprocessor import Preprocessor
from Data.Data import Data
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


d = Data("../Data/")
p = Preprocessor(d)
v = Visualizer()
x = p.load_sales(65, bgn=350, end=450)
y = p.average_smoothing(x)
v.display_sales(x, y)
