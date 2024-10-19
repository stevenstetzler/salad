import numpy as np
import sys
from .primitives import *

class Hough():
    def __init__(self, X, b, dx, dy, tolerance, precompute=True, dtype=np.int32):
        self.X = X
        self.b = b
        self.num_b = self.b.shape[0]
        self.min_x = X[:, 0].min()
        self.min_y = X[:, 1].min()
        self.min_z = X[:, 2].min()
        self.reference_time = self.min_z
        self.dx = dx
        self.dy = dy
        self.precompute = precompute
        self.n = len(X)
        self.tolerance = tolerance
        
        print("creating hough:", self.shape, file=sys.stderr)
        self.array = np.ndarray(self.shape, dtype=dtype)
        
        if self.precompute and (self.num_b * self.n <= 100 * 2**20):
            self.M = transform_to_xy_prime(self.X, self.b, self.reference_time)
            self.bins = digitize_xy(self.M, self.min_x, self.min_y, self.dx, self.dx)
            self.vote_method = "bins"
            self.vote_args = (self.bins,)
        else:
            self.M = None
            self.bins = None
            self.vote_method = "points"
            self.vote_args = (self.X, self.b, self.min_x, self.min_y, self.dx, self.dy, self.reference_time)

    
    def argmax(self):
        return np.unravel_index(self.array.argmax(), self.array.shape)
    
    def max(self):
        return self.array.max()
    
    @property
    def shape(self):
        if not hasattr(self, "_shape"):
            shape = [self.b.shape[0]]
            for i, w in enumerate([self.dx, self.dy]):
                _x = self.X[:, i]
                shape.append(int(((max(_x) - min(_x)) / w) + 1))
            self._shape = tuple(shape)
        return self._shape

    def vote(self, *args, mask=None, use_numba=True, **kwargs):
        if self.vote_method == "points":
            if use_numba:
                _vote = vote_points
            else:
                _vote = vote_points
        elif self.vote_method == "bins":
            if use_numba:
                _vote = vote_bins
            else:
                _vote = vote_bins

        if mask is not None:
            if self.vote_method == "bins":
                vote_args = (self.vote_args[0][:, mask],) + self.vote_args[1:]
            else:
                vote_args = (self.vote_args[0][mask],) + self.vote_args[1:]
        else:
            vote_args = self.vote_args
            
        return _vote(self.array, *vote_args, **kwargs)
    
    def anchor(self, x_idx, y_idx):
        # print(x_idx, y_idx)
        # print(self.dx, self.dy, self.min_x, self.min_y)
        x = ((x_idx + 1/2) * self.dx) + self.min_x
        y = ((y_idx + 1/2) * self.dy) + self.min_y
        return np.array([x, y])
        
    def voters(self, b_idx, x_idx, y_idx):
        m = transform_to_xy_prime(self.X, self.b[b_idx, None], self.reference_time)[0]
        bins = digitize_xy(m[None], self.min_x, self.min_y, self.dx, self.dy)[0]
        mask = (bins[:, 0] == x_idx) & (bins[:, 1] == y_idx)
        return mask
    
    def close(self, b_idx, x_idx, y_idx):            
        return close_to_line(self.X, self.anchor(x_idx, y_idx), self.b[b_idx], self.tolerance, self.reference_time)
    
#     @classmethod
#     def from_catalog(cls, catalog, columns, directions, widths):
#         X = catalog.X(columns=columns, sky_units=sky_units, time_units=time_units)
#         b = directions.b
#         num_b = b.shape[0]
#         shape = [b.shape[0]]
#         for c, w in zip(columns, widths):
#             _x = getattr(catalog, c)
#             shape.append(int(((max(_x) - min(_x))*_x.unit / w).to(u.dimensionless_unscaled) + 1))

#         return cls(tuple(shape), X, b)
    
#     @classmethod
#     def from_search(cls, X, b, widths):
#         num_b = b.shape[0]
#         shape = [b.shape[0]]
#         for i, w in enumerate(width):
#             _x = X[:, i]
#             shape.append(int(((max(_x) - min(_x)) / w) + 1))

#         return cls(tuple(shape), X, b)
    
