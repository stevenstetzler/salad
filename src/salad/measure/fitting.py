import scipy 
import numpy as np

def convert_zp(flux, orig, new):
    return flux * 10**(-2/5*(orig - new))

# can I find the best fitting position?
# by how much can I shift the psf to maximize the flux, snr, or logL?
def logL_position(e, p, shift, ref_zp=31):
    import lsst.afw.image
    eBBox = e.getBBox()
    psfBBox = e.psf.computeImageBBox(p)
    psfBBox.clip(eBBox)

    pc = e.getPhotoCalib()
    zp = pc.instFluxToMagnitude(1)

    mask = np.bitwise_or.reduce(lsst.afw.image.MaskX(e.mask, psfBBox).array.flatten()) # this is the bitwise or across all mask values for pixels that overlap the psf model
    model = lsst.afw.image.ImageD(e.psf.computeImage(p), psfBBox).array
    model = scipy.ndimage.shift(model, shift)
    stamp = lsst.afw.image.ImageD(lsst.afw.image.ImageF(e.image, psfBBox), deep=True).array
    weights = 1/lsst.afw.image.ImageD(lsst.afw.image.ImageF(e.variance, psfBBox), deep=True).array # inverse variance
    c = np.sum(model * stamp * weights, axis=(0, 1)) # Signal; Whidden et al. 2019 Eq. 19 (Psi)
    a = np.sum(model * model * weights, axis=(0, 1)) # Noise; Whidden et al. 2019 Eq. 20 (Phi)
    f = c/a # flux estimate; Whidden et al. 2019 Eq. 22 (alpha_ML)
    sigma = 1/np.sqrt(a) # standard deviation in flux estimate
    snr = c / np.sqrt(a) # signal to noise -- why is this not f / sigma?; Whidden et al. 2019 Eq. 26 (nu_coadd)
    mag = e.getPhotoCalib().instFluxToMagnitude(f)
    sigma_mag_high = mag - e.getPhotoCalib().instFluxToMagnitude(f - sigma)
    sigma_mag_low = e.getPhotoCalib().instFluxToMagnitude(f + sigma) - mag

    logL = -0.5 * np.sum(weights * (f * model - stamp) ** 2)
    return {
        "logL": logL, 
        "a": a,
        "c": c,
        "flux": f, 
        "flux_ref": convert_zp(f, zp, ref_zp),
        "sigma": sigma, 
        "sigma_ref": convert_zp(sigma, zp, ref_zp),
        "SNR": snr,
        "mag": mag,
        "sigma_mag_high": sigma_mag_high,
        "sigma_mag_low": sigma_mag_low,
        "mask": mask,
        "zero_point": zp,
    }

def fit_position(e, p, method="Nelder-Mead"):
    trace = []
    def f(x):
        trace.append(x)
        return -1 * logL_position(e, p, x)['logL']
    
    return trace, scipy.optimize.minimize(
        f, np.array([0, 0]),
        method=method
    )

def logL_trajectory(es, ps, times, trajectory, trajectory_params):
    import lsst.geom
    position_result_keys = list(logL_position(es[0], ps[0], [0, 0]).keys())
    traj_result = {
        "position_" + k: [] for k in position_result_keys
    }
    ra, dec = trajectory(trajectory_params, times) # returns the predicted location at each time in degrees
    if np.any(np.array(dec) > 90) or np.any(np.array(dec) < -90):
        traj_result['logL'] = -np.inf
        return traj_result
    
    ra = list(map(lsst.geom.Angle, ra * np.pi/180))
    dec = list(map(lsst.geom.Angle, dec * np.pi/180))
    skyPoints = [lsst.geom.SpherePoint(r, d) for r, d in zip(ra, dec)]
    pixPoints = [e.wcs.skyToPixel(sp) for e, sp in zip(es, skyPoints)]
    # compute flux/logL for each exposure
    for e, p, p_traj in zip(es, ps, pixPoints):
        _position_result = logL_position(e, p, [p.y - p_traj.y, p.x - p_traj.x])
        for k in position_result_keys:
            traj_result["position_" + k].append(_position_result[k])

    traj_result['logL'] = np.sum(traj_result['position_logL'])
    return traj_result

def fit_trajectory(es, ps, times, trajectory, start=np.array([0, 0]), method="Nelder-Mead"):
    trace = []
    def f(x):
        trace.append(x)
        return -1 * logL_trajectory(es, ps, times, trajectory, x)['logL']
    return scipy.optimize.minimize(
        f, 
        start,
        method=method
    )
