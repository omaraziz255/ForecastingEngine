import sys

from forecaster.data.data import Data
from forecaster.engine.arima import ARIMA
from forecaster.engine.exponential_smooth import ExponentialSmooth
from forecaster.engine.fb_prophet import FBProphet
from forecaster.engine.holt_linear import HoltLinear
from forecaster.engine.moving_average import MovingAverage
from forecaster.engine.naive import Naive
from forecaster.postprocessors.postprocessor import Postprocessor
from forecaster.preprocessors.preprocessor import Preprocessor
from forecaster.visualizer.visualizer import Visualizer


def main(argv):
    d = Data("../Data/")
    p = Preprocessor(d)
    v = Visualizer()
    x = p.load_sales(65, bgn=350, end=450)
    y = p.preprocess_smooth(x)
    v.view_historical_data(x, y)

    x = p.data.sales_data
    y = p.data.selling_prices
    z = p.data.calendar

    n = Naive(p.data)
    mv = MovingAverage(p.data, 30)
    h = HoltLinear(p.data, 30)
    e = ExponentialSmooth(p.data, 30)
    a = ARIMA(p.data, 30)
    ph = FBProphet(p.data, 30)

    methods = [n, mv, h, e, a, ph]
    for m in methods:
        m.predict()

    pp = Postprocessor()
    losses = []
    for m in methods:
        losses.append(pp.postprocess(m.predictions, m.validation))

    names = ["Naive approach", "Moving average", "Holt linear", "Exponential smoothing", "ARIMA", "Prophet"]
    v.visualize_loss(losses, names)

    v.train_val_view(ph, 0, prediction=True)

    return


def entrypoint():
    """ Command line entrypoint. """
    main(sys.argv)
    return


if __name__ == '__main__':
    entrypoint()
