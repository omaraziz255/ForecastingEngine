import numpy as np


class Postprocessor:
    def __init__(self):
        pass

    def postprocess(self, pred, valid): #RMSe
        return np.linalg.norm(pred[:3] - valid.values[:3]) / len(pred[0])
