from ..measure.fitting import logL_position
from ..cutouts.get import times_for
from ..serialize import read, write
from ..cutouts.attach import attach_cutouts_from_exposures
import numpy as np
import astropy.time
import logging

logging.basicConfig()
log = logging.getLogger(__name__)

def zeropoint_correction(orig, zp):
    return 10**(-2/5 * (orig - zp))

def summary(cluster):
    # create a summary of the cluster from the cutouts
    # creates coadds and light curves
    ref_zp = 31
    
    c = list(filter(lambda x : x is not None, cluster.cutouts))
    
    expnum = sorted(cluster.cutouts.keys())
    times = []
    image = []
    variance = []
    mask = []
    flux = []
    sigma = []
    snr = []
    mag = []
    lc_mask = []
    mask_plane_dicts = []
    for e in expnum:
        cutout = cluster.cutouts[e]
        center = cluster.centers[e]
        
        times.append((cutout.visitInfo.date.toAstropy() + astropy.time.TimeDelta(cutout.visitInfo.exposureTime / 2 + 0.5, format='sec')).value)
        
        k = zeropoint_correction(cutout.info.getSummaryStats().zeroPoint, ref_zp)
        image.append(cutout.image.array * k)
        variance.append(cutout.variance.array * k**2)
        mask.append(cutout.mask.array)
        mask_plane_dicts.append(cutout.mask.getMaskPlaneDict())
        
        phot_result = logL_position(cutout, center['pix'], [0, 0], ref_zp=ref_zp)
        flux.append(phot_result['flux_ref'])
        sigma.append(phot_result['sigma_ref'])
        snr.append(phot_result['SNR'])
        mag.append(phot_result['mag'])
        lc_mask.append(phot_result['mask'])

    expnum = np.array(expnum)
    times = np.array(times)
    image = np.array(image)
    variance = np.array(variance)
    mask = np.array(mask)
    flux = np.array(flux)
    sigma = np.array(sigma)
    snr = np.array(snr)
    mag = np.array(mag)
    lc_mask = np.array(lc_mask)
        
    mean_coadd = np.mean(image, axis=0)
    sum_coadd = np.sum(image, axis=0)
    median_coadd = np.median(image, axis=0)
    weighted_coadd = (np.sum((image / variance), axis=0) / np.sum(1/variance, axis=0))
    
    return {
        "expnum": expnum,
        "time": times,
        "image": image,
        "variance": variance,
        "mask": mask,
        "mask_plane_dict": mask_plane_dicts,
        "coadd": {
            "sum": sum_coadd,
            "mean": mean_coadd,
            "median": median_coadd,
            "weighted": weighted_coadd,
        },
        "light_curve": {
            "flux": flux,
            "sigma": sigma,
            "snr": snr,
            "mag": mag,
            "mask": lc_mask,
        }
    }

def main():
    import argparse
    import sys
    import os

    parser = argparse.ArgumentParser(prog=__name__)

    parser.add_argument("clusters", nargs='+', type=str)
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--width", type=int, default=50)
    parser.add_argument("--height", type=int, default=50)
    args = parser.parse_args()

    for path in args.clusters:
        assert(os.path.exists(path))

    images = read(args.images)
    exposures = {}
    for i, image in enumerate(images):
        exposures[image.expnum] = image.exposure

    for path in args.clusters:
        if not os.path.exists(path):
            log.warn("path %s does not exist", path)
            continue
        try:
            clusters = read(path)
        except Exception as e:
            log.warn("could not read %s: %s", path, str(e))
            continue
            
        attach_cutouts_from_exposures(clusters, exposures, args.width, args.height)
        for i in clusters:
            cluster = clusters[i]
            cluster.summary = summary(cluster)
            cluster.cutouts = None

        write(
            clusters, 
            os.path.join(
                os.path.dirname(path), 
                os.path.basename(path).replace(".pkl", "_summary.pkl")
            )
        )
        del clusters

if __name__ == "__main__":
    main()