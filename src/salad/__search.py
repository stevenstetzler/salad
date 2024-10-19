import logging
import sys
import numpy as np
import numba
import sklearn
from sklearn.covariance import MinCovDet
from scipy.stats import chi
import matplotlib.pyplot as plt
import argparse
import astropy
import astropy.units as u
from astropy.table import Table
import time

logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(levelname)s:%(filename)s:%(lineno)s - %(funcName)5s()] %(message)s")
# logger.setLevel(logging.DEBUG)

def get_directions_for_search(v_min, v_max, phi_min, phi_max, dx, dt):
    v_ra = np.arange(-v_max, v_max + (dx/dt), (dx/dt))
    v_dec = np.arange(-v_max, v_max + (dx/dt), (dx/dt))
    v_ra_m, v_dec_m = np.meshgrid(v_ra, v_dec)
    b = np.vstack([v_ra_m.flatten(), v_dec_m.flatten()]).T
    _v = (b**2).sum(axis=1)**0.5
    _phi = np.arctan2(b[:, 1], b[:, 0])

    v_mask = (_v >= v_min) & (_v <= v_max)
    phi_mask = (_phi >= phi_min) & (_phi <= phi_max)
    b_m = b.reshape(len(v_ra), len(v_dec), 2)
    adjacent_dx_width = (b_m - np.roll(np.roll(b_m, 1, axis=0), 1, axis=1))[1:, 1:] * dt/dx
    assert(np.allclose(adjacent_dx_width[~np.isnan(adjacent_dx_width)], 1)) # all adjacent velocities are within 1 dx over timespan of dt
    return b[v_mask & phi_mask]


def _make_bins(x, dx):
    _range = [x.min(), x.max()]
    nbins = int((_range[1] - _range[0])/dx + 0.5) + 1
    bins, step = np.linspace(x.min(), x.max() + dx, nbins, retstep=True, endpoint=True)
    return bins, step

def make_bins(X, dx, dy):
    bins_x, step_x = _make_bins(X[:, 0], dx)
    bins_y, step_y = _make_bins(X[:, 1], dy)
    return (bins_x, step_x), (bins_y, step_y)

@numba.njit()
def xyz_to_xy_prime(x: np.ndarray, b: np.ndarray, reference_time: np.float64):
    vx = b[0]
    vy = b[1]
    x_prime = x[0] - vx * (x[2] - reference_time)
    y_prime = x[1] - vy * (x[2] - reference_time)
    return x_prime, y_prime
                
@numba.njit(parallel=True)
def transform_to_xy_prime(X: np.ndarray, b: np.ndarray, reference_time: np.float64):
    num_b = b.shape[0]
    n = X.shape[0]
    M = np.zeros((num_b, n, 2))
    for i in numba.prange(num_b):
        for j in range(n):
            M[i, j] = xyz_to_xy_prime(X[j], b[i], reference_time)
    return M

@numba.njit(parallel=True)
def vote_hough_X(X: np.ndarray, b: np.ndarray, x_min, y_min, dx, dy, reference_time, hough: np.ndarray, coef: np.float64=1):
    n, d = X.shape
    num_b = b.shape[0]
    for b_idx in numba.prange(num_b): # for each direction
        if b_idx < num_b:
            b_i = b[b_idx]
            for i in range(n): # for each point
                x_prime, y_prime = xyz_to_xy_prime(X[i], b_i, reference_time)
                x_idx, y_idx = digitize_point(np.array([x_prime, y_prime]), x_min, y_min, dx, dy)
                if x_idx < hough.shape[1] and y_idx < hough.shape[2]:
                    hough[b_idx, x_idx, y_idx] += coef


@numba.njit(parallel=True)
def vote_hough_bins(bins: np.ndarray, hough: np.ndarray, coef: np.float64=1):
    num_b, n, _ = bins.shape
    for b_idx in numba.prange(num_b): # for each direction
        for i in range(n): # for each point
            if b_idx < num_b:
                x_idx = bins[b_idx, i, 0] # bin location for that point
                y_idx = bins[b_idx, i, 1]
                if x_idx < hough.shape[1] and y_idx < hough.shape[2]:
                    hough[b_idx, x_idx, y_idx] += coef

@numba.njit(parallel=True)
def transform_v_phi_to_b(v: np.ndarray, phi: np.ndarray):
    num_v = v.shape[0]
    num_phi = phi.shape[0]
    
    b = np.zeros((num_v * num_phi, 2))
    for i in numba.prange(num_v):
        for j in range(num_phi):
            b[i * num_phi + j, 0] = v[i] * np.cos(phi[j])
            b[i * num_phi + j, 1] = v[i] * np.sin(phi[j])
    return b

def hough_argmax(hough: np.ndarray):
    return np.unravel_index(hough.argmax(), hough.shape)

@numba.njit()
def digitize_point(p, min_x, min_y, dx, dy):
    return int((p[0] - min_x)/dx), int((p[1] - min_y)/dy)

@numba.njit(parallel=True)
def digitize_xy(points, min_x, min_y, dx, dy):
    bins = np.zeros(points.shape, dtype=np.int32)
#     bins_y = np.zeros(points.shape[0])
    for i in numba.prange(points.shape[0]):
        for j in range(points.shape[1]):
            x, y = digitize_point(points[i, j], min_x, min_y, dx, dy)
            bins[i, j, 0] = x
            bins[i, j, 1] = y
    return bins

import scipy 

def mean_and_covar(z, robust=True):
    if robust:
        mcd = MinCovDet()
        mcd.fit(z)
        _mu = mcd.location_
        _covar = mcd.covariance_
    else:
        _mu = np.mean(z, axis=0)
        _covar = (z - _mu).T @ (z - _mu)
    return _mu, _covar
        
def regression(x, y, robust=True):
    logger.debug("solving regression problem x=%s -> y=%s", x.shape, y.shape)
    z = np.hstack([x, y])
    shift_z = - z.min(axis=0) / (z.max(axis=0) - z.min(axis=0))
    scale_z = 1 / (z.max(axis=0) - z.min(axis=0))
    A = np.diag(scale_z)
    inv_A = np.diag(1/scale_z)
    inv_A_T = inv_A
    b = shift_z
    z = z @ A + b

    _mu, _covar = mean_and_covar(z, robust=robust)
    
    mu = (_mu - b) @ inv_A
    covar = inv_A_T @ _covar @ inv_A

    logger.debug("mu=%s", mu)
    logger.debug("covar=%s", covar)

    mu_x = mu[:1]
    mu_y = mu[1:]

    sigma_xx = covar[:1, :1]
    sigma_xy = covar[:1, 1:]
    sigma_yy = covar[1:, 1:]

    beta = scipy.linalg.pinvh(sigma_xx) @ sigma_xy
    sigma_e = sigma_yy - beta.T @ sigma_xx @ beta
    logger.debug("beta=%s", beta)
    logger.debug("sigma_e=%s", sigma_e)
    logger.debug("sigma_xx=%s", sigma_xx)
    # if np.any(sigma_e < 1e-12):
    #     logger.warning("clipping small (or negative values) in regression covariance matrix")
    #     sigma_e = np.clip(sigma_e, 1e-12, None)
    # if np.any(sigma_xx < 1e-12):
    #     logger.warning("clipping small (or negative values) in clustering covariance matrix")
    #     sigma_xx = np.clip(sigma_xx, 1e-12, None)

    alpha = mu_y - beta.T @ mu_x

    y_hat = x @ beta + alpha
    r = y - y_hat

    d_r = ((r @ scipy.linalg.pinvh(sigma_e) * r).sum(1))**0.5
    d_x = (((x - mu_x) @ scipy.linalg.pinvh(sigma_xx) * (x - mu_x)).sum(1))**0.5
    logger.debug("d_r=%s", d_r)
    logger.debug("d_x=%s", d_x)

    if np.any(np.isnan(d_r)):
        return None

    outliers_r = d_r > scipy.stats.chi.ppf(0.975, df=y.shape[1])
    outliers_x = d_x > scipy.stats.chi.ppf(0.975, df=x.shape[1])
    return beta, alpha, outliers_r, d_r, outliers_x, d_x, sigma_e, sigma_xx

def close_to_line(X, anchor, direction, dx, reference_time):
    M = transform_to_xy_prime(X, np.atleast_2d(direction), reference_time)[0]
    d = (anchor - M)
    d_norm = np.sum(d**2, axis=1)**0.5
    return d_norm <= dx

class RegressionResult():
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
    def pred(self, t, reference_time):
        return pred(t, self.a, self.b, reference_time)

class SearchResult():
    def __init__(self, search, n, b_idx, x_idx, y_idx):
        self.n = n
        self.b_idx = b_idx
        self.x_idx = x_idx
        self.y_idx = y_idx
        self.search = search

    @property
    def voters(self):
        self.search.get_voters(self.b_idx, self.x_idx, self.y_idx)
        
    def close(self, tol=None):
        if tol is None:
            tol = self.search.dx
        return close_to_line(self.search.X, )

    def __reduce__(self):
        return type(self), (
            self.search, self.n, self.b_idx, self.x_idx, self.y_idx
        )

class Search():
    def __init__(self, catalog, b, dx, reference_time, precompute=True):
        self.catalog = catalog
        self.X = self.catalog.X
        self.X[:, 0] *= 180 / np.pi
        self.X[:, 1] *= 180 / np.pi
        self.dx = dx
        self.reference_time = reference_time
        self.precompute = precompute
        
        self.n = self.X.shape[0]
        self.b = np.array(b)
        self.num_b = self.b.shape[0]
        
        self.min_x, self.max_x = self.X[:, 0].min(), self.X[:, 0].max()
        self.min_y, self.max_y = self.X[:, 1].min(), self.X[:, 1].max()
        self.num_x = int((self.max_x - self.min_x)/self.dx + 1)
        self.num_y = int((self.max_y - self.min_y)/self.dx + 1)
        
        self.hough = np.zeros((self.num_b, self.num_x, self.num_y), dtype=np.int32)
        self.mask = np.ones(self.n, dtype=bool)

        if self.precompute and (self.num_b * self.n <= 2**30):
            logger.info("precomputing transformed coordinates")
            self.M = transform_to_xy_prime(self.X, self.b, self.reference_time)
            self.bins = digitize_xy(self.M, self.min_x, self.min_y, self.dx, self.dx)
            self.vote_method = vote_hough_bins
            self.vote_args = (self.bins, self.hough,)
        else:
            self.M = None
            self.bins = None
            self.vote_method = vote_hough_X
            self.vote_args = (self.X, self.b, self.min_x, self.min_y, self.dx, self.dx, self.reference_time, self.hough)
            
    def __reduce__(self):
        return type(self), (
            self.catalog, self.b, self.dx, self.reference_time
        )

    def vote(self, mask=None, **kwargs):
        if mask is not None:
            if len(self.vote_args[0].shape) == 3:
                vote_args = (self.vote_args[0][:, mask],) + self.vote_args[1:]
            else:
                vote_args = (self.vote_args[0][mask],) + self.vote_args[1:]
        else:
            vote_args = self.vote_args
        self.vote_method(*vote_args, **kwargs)
        
    def get_voters(self, b_idx, x_idx, y_idx):
        logger.debug("getting voters for (%d, %d, %d)", b_idx, x_idx, y_idx)
        if self.bins is not None:
            bins = self.bins[b_idx]
        elif self.M  is not None:
            bins = digitize_xy(self.M[b_idx][None], self.min_x, self.min_y, self.dx, self.dx)[0]
        else:
            m = transform_to_xy_prime(self.X, self.b[b_idx, None], self.reference_time)[0]
            bins = digitize_xy(m[None], self.min_x, self.min_y, self.dx, self.dx)[0]
#             bin_nums_x = ((m[:, 0] - self.bins_x[0])/self.step_x).astype(np.int32)
#             bin_nums_y = ((m[:, 1] - self.bins_y[0])/self.step_y).astype(np.int32)
        mask = (bins[:, 0] == x_idx) & (bins[:, 1] == y_idx)
        return mask

    def refine(self, X, robust=True):
        regression_result = regression(X[:, 2][:, None], X[:, :2], robust=robust)
        if regression_result is not None:
            slope, intercept, outliers_r, d_r, outliers_x, d_x, regression_covar, sigma_xx = regression_result
            return np.atleast_1d(self.reference_time) @ slope + intercept, slope[0]
        else:
            return None, None
    
#     def refine_robust(self, X):
#         mcd = MinCovDet()
#         mcd.fit(X)
#         mu, sigma = mcd.location_, mcd.covariance_
#         ev, evec = np.linalg.eig(sigma)
#         return mu, -evec[:, 0]
        
    def get_top_line(self, robust=True, refine_direction=True, refine_anchor=True):
        b_idx, x_idx, y_idx = hough_argmax(self.hough)
        logger.debug("hough max at (%d, %d, %d)", b_idx, x_idx, y_idx)
        x = ((x_idx + 1/2) * self.dx) + self.min_x
        y = ((y_idx + 1/2) * self.dx) + self.min_y

        hough_anchor = np.array([x, y])
        hough_dir = self.b[b_idx]
        voters = self.get_voters(b_idx, x_idx, y_idx) & self.mask
        close = close_to_line(self.X, hough_anchor[None], hough_dir, self.dx, self.reference_time) & self.mask
        logger.debug("there are %d close points", close.sum())
        if refine_anchor and not refine_direction:
            # get the (robust) mean of the data points in the x' y' space
            xy = transform_to_xy_prime(self.X[close], hough_dir[None], self.reference_time)[0]
            _mu, _ = mean_and_covar(xy, robust=robust)
            refined_anchor = _mu[:2]
            refined_dir = hough_dir.copy()
        elif refine_anchor and refine_direction:
            # get the (robust) slope and (robust) intercept of a regression model
            # the new anchor is the position at the reference time using the (robust) slope and (robust) intercept
            # the new direction is the (robust) slope
            refined_anchor, refined_dir = self.refine(self.X[close], robust=robust)
            if refined_anchor is None:
                raise Exception("error in refinement")
        elif not refine_anchor and refine_direction:
            # the new direction is the (robust) slope
            # use the old anchor
            # undefined?
            raise Exception("can't refine direction and not the anchor")
        else:
            # do nothing
            refined_anchor = hough_anchor.copy()
            refined_dir = hough_dir.copy()
        
        refined = close_to_line(self.X, refined_anchor[None], refined_dir[None], self.dx, self.reference_time) & self.mask
#         if refine_anchor or refine_direction:
#             refined_anchor, refined_dir = self.refine(self.X[close], robust=robust)
#             print(refined_anchor.shape, refined_dir.shape)
#             if not refine_anchor:
#                 refined_anchor = hough_anchor
#             if not refine_direction:
#                 refined_dir = hough_dir
#             refined = close_to_line(self.X, refined_anchor, refined_dir, self.dx, self.reference_time) & self.mask
#         else:
#             refined_dir = None
#             refined = close.copy()
#         hough_anchor = hough_anchor[0] # 2d -> 1d
#         refined_anchor = refined_anchor[0] # 2d -> 1d
        return self.hough.max(), (b_idx, x_idx, y_idx), (hough_anchor, hough_dir, voters), close, (refined_anchor, refined_dir, refined)
        
        
    def pop(self, robust=True, refine_direction=True, refine_anchor=True):
        votes, (b_idx, x_idx, y_idx), (hough_anchor, hough_dir, voters), close, (refined_anchor, refined_dir, refined) = self.get_top_line(robust=robust, refine_direction=refine_direction, refine_anchor=refine_anchor)
        
        logger.debug("subtracting %d points", (self.mask & refined).sum())
        self.vote(mask=self.mask & refined, coef=-1)
#         if self.bin_nums is not None:
#             vote_hough_bins(self.bin_nums[:, self.mask & refined], self.hough, coef=-1)
#         elif self.M is not None:
#             vote_hough_M(self.M[:, self.mask & refined], self.bins, self.hough, coef=-1)
#         else:
#             vote_hough_X(self.X[self.mask & refined], self.b, self.bins, self.hough, coef=-1)
        self.mask = self.mask & (~refined) # remove points from those under consideration
        return votes, (b_idx, x_idx, y_idx), (hough_anchor, hough_dir, voters), close, (refined_anchor, refined_dir, refined)

def pred(t, a, b, reference_time):
    t = np.atleast_2d(t) - reference_time
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    return t.T @ b + a

def plot_result(h, result, title, fig_and_axd=None):
    votes, (b_idx, x_idx, y_idx), (hough_anchor, hough_dir, voters), close, (refined_anchor, refined_dir, refined) = result
#     hough_anchor_M = xyz_to_xy_prime(hough_anchor, hough_dir, h.reference_time)
#     refined_anchor_M = xyz_to_xy_prime(refined_anchor, refined_dir, h.reference_time)

    if fig_and_axd is None:
        fig, axd = plt.subplot_mosaic([['tl', 'tr'], ['bl', 'br']], dpi=150, figsize=[8, 8])

    plt.sca(axd['tl'])
    M = transform_to_xy_prime(h.X[close], h.b[b_idx, None], h.reference_time)[0]
    plt.scatter(M[:, 0], M[:, 1])
    M = transform_to_xy_prime(h.X[voters], h.b[b_idx, None], h.reference_time)[0]
    plt.scatter(M[:, 0], M[:, 1])
    c = plt.Circle(hough_anchor, radius=h.dx, fill=False, color="k")
    plt.gca().add_patch(c)
    plt.scatter(hough_anchor[0], hough_anchor[1], color="k")
    plt.axvline(x_idx * h.dx + h.min_x)
    plt.axvline((x_idx + 1) * h.dx + h.min_x)
    plt.axhline((y_idx + 0) * h.dx + h.min_y)
    plt.axhline((y_idx + 1) * h.dx + h.min_y)
    plt.xlabel("x'")
    plt.ylabel("y'")

    plt.sca(axd['bl'])
    # M = transform_to_xy_prime(h.X[close], refined_dir[:, None])[0]
    # plt.scatter(M[:, 0], M[:, 1])
    M = transform_to_xy_prime(h.X[refined], refined_dir[:, None], h.reference_time)[0]
    plt.scatter(M[:, 0], M[:, 1])
    M = transform_to_xy_prime(h.X[voters], refined_dir[:, None], h.reference_time)[0]
    plt.scatter(M[:, 0], M[:, 1])
    c = plt.Circle(refined_anchor, radius=h.dx, fill=False, color="k")
    plt.gca().add_patch(c)
    plt.scatter(refined_anchor[0], refined_anchor[1], color="C3")
    plt.axvline(x_idx * h.dx + h.min_x)
    plt.axvline((x_idx + 1) * h.dx + h.min_x)
    plt.axhline((y_idx + 0) * h.dx + h.min_y)
    plt.axhline((y_idx + 1) * h.dx + h.min_y)
    plt.xlabel("x'")
    plt.ylabel("y'")

    pred_X = pred(h.X[voters, 2], hough_anchor, hough_dir, h.reference_time)
    # plt.figure(dpi=150, figsize=[4, 4])

    plt.sca(axd['tr'])
    plt.scatter(h.X[close, 0], h.X[close, 1], label="close")
    plt.scatter(h.X[voters, 0], h.X[voters, 1], label="voter")
    # plt.scatter(X[close, 2], X[close, 0])
    plt.plot(pred_X[:, 0], pred_X[:, 1], color="k", label="guess")
    plt.xlabel("RA")
    plt.ylabel("Dec")
    plt.legend()

    plt.sca(axd['br'])
    # plt.scatter(h.X[close, 0], h.X[close, 1], label="close")
    plt.scatter(h.X[refined, 0], h.X[refined, 1], label="refined")
    plt.scatter(h.X[voters, 0], h.X[voters, 1], label="voter")
    pred_X = pred(h.X[refined, 2], refined_anchor, refined_dir, h.reference_time)
    plt.plot(pred_X[:, 0], pred_X[:, 1], color="k", label="final")
    plt.xlabel("RA")
    plt.ylabel("Dec")
    plt.legend()
    # plt.plot(pred_X[:, 2], pred_X[:, 0] + dx)
    # plt.plot(pred_X[:, 2], pred_X[:, 0] - dx)
    fig.align_ylabels()
    fig.suptitle(title)
    fig.tight_layout()
    plt.close()
    return fig, axd
