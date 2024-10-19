from sklearn.covariance import MinCovDet
import scipy
import numpy as np
import sys
from .serialize import Serializable

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

class RegressionResult(Serializable):
    beta = None
    alpha = None
    outliers_r = None
    d_r = None
    outliers_x = None
    d_x = None
    sigma_e = None
    sigma_xx = None
    
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

#     logger.debug("mu=%s", mu)
#     logger.debug("covar=%s", covar)

    mu_x = mu[:1]
    mu_y = mu[1:]

    sigma_xx = covar[:1, :1]
    sigma_xy = covar[:1, 1:]
    sigma_yy = covar[1:, 1:]

    beta = scipy.linalg.pinvh(sigma_xx) @ sigma_xy
    sigma_e = sigma_yy - beta.T @ sigma_xx @ beta
#     logger.debug("beta=%s", beta)
#     logger.debug("sigma_e=%s", sigma_e)
#     logger.debug("sigma_xx=%s", sigma_xx)
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
#     logger.debug("d_r=%s", d_r)
#     logger.debug("d_x=%s", d_x)

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
        

