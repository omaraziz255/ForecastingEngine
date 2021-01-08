import numpy as np


class Postprocessor:
    def __init__(self):
        pass

    def compute_loss(self, pred, valid):
        return np.linalg.norm(pred[:3] - valid.values[:3]) / len(pred[0])
