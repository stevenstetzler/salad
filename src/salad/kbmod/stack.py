import torch
import argparse
import sys
from pathlib import Path
import numpy as np
import astropy.units as u
import logging
from scipy.signal import convolve2d
from time import time
import lsst.afw.math as afwMath
import lsst.meas.algorithms as measAlg

from ..serialize import read
from ..directions import SearchDirections

logging.basicConfig()
log = logging.getLogger(__name__)


def main():
    t1 = time()
    log.info("start %s %s", __name__, t1)

    import argparse
    parser = argparse.ArgumentParser(prog=__name__)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--velocity", required=True, nargs=2, type=float)
    parser.add_argument("--angle", required=True, nargs=2, type=float)
    parser.add_argument("--dx", required=True, type=float)
    parser.add_argument("--min-snr", default=10.0, type=float)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-results", default=100, type=int)

    args = parser.parse_args()

    if args.use_gpu and not torch.cuda.is_available():
        raise RuntimeError("No GPU is available and --use-gpu was passed")
    
    device = torch.device("cuda:0" if args.use_gpu and torch.cuda.is_available() else "cpu")

    np_dtype = getattr(np, args.dtype)
    torch_dtype = getattr(torch, args.dtype)
    planes_to_mask = ["BAD", "CROSSTALK", "EDGE", "SAT", "STREAK", "SUSPECT", "DETECTED"]

    images = read(args.images)
    
    log.info("start loading image times %s", time() - t1)
    mjds = np.zeros(len(images))
    for i, image in enumerate(images):
        mjds[i] = image.mjd_mid

    log.info("end loading image times %s", time() - t1)

    idx = np.argsort(mjds)
    
    mjds = mjds[idx]

    log.info("start loading/reprojecting/convolving images %s", time() - t1)

    reference_exposure = images[idx[0]].exposure
    reference_wcs = reference_exposure.getWcs()
    reference_bbox = reference_exposure.getBBox()
    reference_psf = reference_exposure.getPsf()

    warper = afwMath.Warper("lanczos5")
    warp = lambda x : warper.warpExposure(reference_wcs, x, destBBox=reference_bbox)

    detection = measAlg.SourceDetectionTask()

    psis = []
    phis = []
    # masks = []
    seeing = []
    offsets_x = []
    offsets_y = []
    for j, i in enumerate(idx):
        # if j > 3:
        #     break
        
        image = images[i]
        exposure = image.exposure
        psf = exposure.getPsf()

        log.info("start warp %s", time() - t1)
        # warped = exposure
        warped = warp(exposure)
        bbox = warped.getBBox()
        log.info("end warp %s", time() - t1)

        # a = warped.image.Factory(bbox)
        # a.array = warped.image / warped.variance
        
        # sigma = psf.computeShape(psf.getAveragePosition()).getDeterminantRadius()
        # kWidth = detection.calculateKernelSize(sigma)
        # gaussFunc = afwMath.GaussianFunction1D(sigma)
        # gaussKernel = afwMath.SeparableKernel(kWidth, kWidth, gaussFunc, gaussFunc)

        # psi = warped.image.Factory(bbox)
        # afwMath.convolveImage(psi, a, gaussKernel, afwMath.ConvolutionControl())

        # phi = warped.variance.Factory(bbox)
        # afwMath.convolveImage(phi, warped.variance, gaussKernel, afwMath.ConvolutionControl())

        # phi = maskedImage.Factory(maskedImage.getBBox())

        # t2 = time()
        log.info("start convolve %s", time() - t1)
        # t1 = t2
        result = detection.convolveImage(warped.getMaskedImage(), reference_psf)#psf)
        convolved = result.middle
        psi = torch.tensor(convolved.image.array.astype(np_dtype))
        phi = torch.tensor(convolved.variance.array.astype(np_dtype))
        
        mask = convolved.mask.array
        bad = np.zeros(convolved.mask.array.shape)
        maskPlaneDict = convolved.mask.getMaskPlaneDict()
        for m in planes_to_mask:
            bad += (mask >> maskPlaneDict[m]) & 1
        
        # print("number of bad pixels", (bad != 0).sum())
        to_mask = torch.where(torch.tensor(bad) != 0)
        psi[to_mask] = 0.0
        phi[to_mask] = 0.0
        # masks.append(bad)
        # t2 = time()
        log.info("end convolve %s", time() - t1)
        # t1 = t2

        # _image = torch.tensor(warped.getImage().array.astype(np_dtype)[256:512, 256:512])
        # image = torch.zeros(1, 1, _image.shape[0], _image.shape[1], dtype=torch_dtype)
        # image[0, 0, :, :] = _image
        # image.to(device)

        # _variance = torch.tensor(warped.getVariance().array.astype(np_dtype)[256:512, 256:512])
        # variance = torch.zeros(1, 1, _variance.shape[0], _variance.shape[1], dtype=torch_dtype)
        # variance[0, 0, :, :] = _variance
        # variance.to(device)

        # _kernel = torch.tensor(psf.computeImage(bbox.getCenter()).array.astype(np_dtype))[10:31, 10:31]
        # kernel = torch.zeros(1, 1, _kernel.shape[0], _kernel.shape[1], dtype=torch_dtype)
        # kernel[0, 0, :, :] = _kernel
        # kernel.to(device)

        # t2 = time()
        # log.info("start convolve %s", t2 - t1)
        # t1 = t2
        # psis.append(torch.conv2d(image / variance, kernel, padding='same')[0, 0])
        # phis.append(torch.conv2d(variance, kernel * kernel, padding='same')[0, 0])
        # t2 = time()
        # log.info("end convolve %s", t2 - t1)
        # t1 = t2

        # psis.append(torch.tensor(warped.image.array.astype(np_dtype)))
        # phis.append(torch.tensor(warped.variance.array.astype(np_dtype)))
        offsets_x.append(convolved.getBBox().getMinX())
        offsets_y.append(convolved.getBBox().getMinY())
        psis.append(psi.to(device))
        phis.append(phi.to(device))
        # print(psis[-1].shape, phis[-1].shape)
        seeing.append(psf.computeShape(bbox.getCenter()).getDeterminantRadius())


    # t2 = time()
    log.info("end loading/reprojecting/convolving images %s", time() - t1)
    # t1 = t2

    offsets_x = torch.tensor(offsets_x).to(device)
    offsets_y = torch.tensor(offsets_y).to(device)
    seeing = torch.tensor(seeing).to(device)

    ref_time = mjds[0]
    dmjds = mjds - ref_time

    pixel_scale = (0.263 * u.arcsec / u.pixel)
    dt = (max(mjds) - min(mjds))*u.day
    vmin = args.velocity[0] * u.pixel/u.day
    vmax = args.velocity[1] * u.pixel/u.day
    phimin = args.angle[0] * u.deg
    phimax = args.angle[1] * u.deg
    dx = args.dx * u.pixel

    directions = SearchDirections([vmin, vmax], [phimin, phimax], dx, dt)

    log.info("searching %d directions", len(directions.b))

    # t2 = time()
    log.info("start search %s", time() - t1)
    # t1 = t2
    results = {}
    for i, direction in enumerate(directions.b.value): # for each search direction
        psi_stack = torch.zeros_like(psis[0]).to(device)
        phi_stack = torch.zeros_like(phis[0]).to(device)
        # mask_stack = torch.zeros_like(masks[0]).astype(torch.bool)
        for dmjd, psi, phi in zip(dmjds, psis, phis): # for each image
            # shift
            shift = (-round(dmjd*direction[0]), -round(dmjd*direction[1])) # compute pixel shift
            # print(shift)
            # stack
            psi_stack += psi.roll(shift, dims=(0, 1))
            phi_stack += phi.roll(shift, dims=(0, 1))
            # mask_stack |= mask_stack

        snr = psi_stack / phi_stack**0.5
        snr = torch.nan_to_num(snr, -1.0)

        idx_x, idx_y = torch.where(snr > args.min_snr)
        for x, y in zip(idx_x, idx_y):
            s = float(snr[x, y])
            results[s] = results.get(s, (direction[0], direction[1], int(x), int(y)))
    
    log.info("start sort results %s", time() - t1)
    keys = sorted(results.keys(), reverse=True)[:args.max_results]
    log.info("end sort results %s", time() - t1)

    log.info("start print results %s", time() - t1)
    for s in keys:
        vx, vy, x, y = results[s]
        print(vx, vy, int(x + offsets_x[0]), int(y + offsets_y[0]), float(s))
    log.info("end print results %s", time() - t1)
    # t2 = time()
    # t1 = t2
    log.info("end %s %s", __name__, time() - t1)

if __name__ == "__main__":
    main()
