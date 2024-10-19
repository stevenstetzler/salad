import numpy as np
import astropy.table

class Line(object):
    alpha = None 
    beta = None
    offset = 0
    
    def __init__(self, **kwargs):
        for k in kwargs:
            setattr(self, k, kwargs[k])
    
    def predict(self, time):
        beta = np.atleast_2d(self.beta)
        if isinstance(time, np.ndarray) and hasattr(time, "unit") and not hasattr(self.offset, "unit"):
            time = astropy.table.Column(time)
        t = (np.atleast_2d(time) - self.offset)
        alpha = np.atleast_2d(self.alpha)
        return t.T @ beta + alpha
