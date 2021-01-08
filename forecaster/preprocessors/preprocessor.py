#!/usr/bin/env python
# coding: utf8

import numpy as np
from forecaster.data.data import Data
import pywt


def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


class Preprocessor:
    def __init__(self, d: Data):
        self.data = d

    def load_sales(self, sales_idx, bgn=None, end=None):
        sales = self.data.sales_data
        ids = sorted(list(set(sales['id'])))
        d_cols = [c for c in sales.columns if 'd_' in c]
        return sales.loc[sales['id'] == ids[sales_idx]].set_index('id')[d_cols].values[0][slice(bgn, end)]

    def denoise_signal(self, x, wavelet='db4', level=1):
        coeff = pywt.wavedec(x, wavelet, mode='per')
        sigma = (1 / 0.6745) * maddest(coeff[-level])
        uthresh = sigma * np.sqrt(2 * np.log(len(x)))
        coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
        return pywt.waverec(coeff, wavelet, mode='per')

    def average_smoothing(self, x, kernel_size=3, stride=1):
        sample = []
        start = 0
        end = kernel_size
        while end <= len(x):
            start += stride
            end += stride
            sample.extend(np.ones(end - start) * np.mean(x[start:end]))
        return np.array(sample)
