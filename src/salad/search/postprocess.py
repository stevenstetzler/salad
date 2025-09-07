import logging
import numpy as np
from sklearn.covariance import MinCovDet
import scipy
import sys
import astropy.table

logging.basicConfig()
log = logging.getLogger(__name__)

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


class RegressionResult(object):
    beta = None
    alpha = None
    outliers_r = None
    d_r = None
    outliers_x = None
    d_x = None
    sigma_e = None
    sigma_xx = None


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
#     logger.debug("solving regression problem x=%s -> y=%s", x.shape, y.shape)
    z = np.hstack([x, y])
    if len(z) == 0:
        print(f"no data available", file=sys.stderr)
        return None
    
    shift_z = - z.min(axis=0) / (z.max(axis=0) - z.min(axis=0))
    scale_z = 1 / (z.max(axis=0) - z.min(axis=0))
    if np.any(np.isnan(scale_z)):
        print(f"data scaling has NaN values; the input must contain duplicate values", file=sys.stderr)
        return None
    if np.any(np.isinf(scale_z)):
        print(f"data scaling has infinity values; the input must contain duplicate values", file=sys.stderr)
        return None
    
    A = np.diag(scale_z)
    inv_A = np.diag(1/scale_z)
    inv_A_T = inv_A
    b = shift_z
    z = z @ A + b

    _mu, _covar = mean_and_covar(z, robust=robust)
    
    mu = (_mu - b) @ inv_A
    covar = inv_A_T @ _covar @ inv_A

    mu_x = mu[:1]
    mu_y = mu[1:]

    sigma_xx = covar[:1, :1]
    sigma_xy = covar[:1, 1:]
    sigma_yy = covar[1:, 1:]

    beta = scipy.linalg.pinvh(sigma_xx) @ sigma_xy
    sigma_e = sigma_yy - beta.T @ sigma_xx @ beta

    alpha = mu_y - beta.T @ mu_x

    y_hat = x @ beta + alpha
    r = y - y_hat

    d_r = ((r @ scipy.linalg.pinvh(sigma_e) * r).sum(1))**0.5
    d_x = (((x - mu_x) @ scipy.linalg.pinvh(sigma_xx) * (x - mu_x)).sum(1))**0.5

    if np.any(np.isnan(d_r)):
        print(f"regression returned NaN results", file=sys.stderr)
        return None

    outliers_r = d_r > scipy.stats.chi.ppf(0.975, df=y.shape[1])
    outliers_x = d_x > scipy.stats.chi.ppf(0.975, df=x.shape[1])
    result = RegressionResult()
    result.beta = beta
    result.alpha = alpha
    result.outliers_r = outliers_r
    result.d_r = d_r
    result.outliers_x = outliers_x
    result.d_x = d_x
    result.sigma_e = sigma_e
    result.sigma_xx = sigma_xx
    return result

def refine(points):
    x, y = points[:, 2][:, None], points[:, :2]
    log.info(f"refining cluster with {len(points)} points")
    if len(points) < 2:
        log.warn("cluster has too few points to fit a line")
        return None
    try:
        regression_result = regression(x, y)
        if regression_result is None:
            log.warn("regression on points in cluster failed")
            return None
    except Exception as e:
        log.exception(e)
        return None
    regression_error = (np.diag(regression_result.sigma_e)**2).sum()
    log.debug(f"regression error {regression_error}")
    outliers = regression_result.outliers_r
    inliers = ~outliers
    result = regression_result
    outliers = result.outliers_r
    inliers = ~outliers
    log.info(f"refined cluster has {inliers.sum()} inliers")

    line = Line(
        alpha=result.alpha,
        beta=result.beta,
    )
    return dict(
        result=result,
        line=line
    )


def gather(result : RegressionResult, ra, dec, time, threshold):
    """
    gather all points within some threshold of a line
    """
    y_pred = np.dot(time[:, None], result.beta) + result.alpha
    residuals = (
        np.array([ra, dec]).T - 
        y_pred
    )
    distance = (residuals**2).sum(axis=1)**0.5
    mask = distance < threshold
    log.info("gathered %s points", mask.sum())
    return mask