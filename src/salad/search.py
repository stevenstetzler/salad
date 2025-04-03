from .serialize import read, write
from .hough import Hough
from .directions import SearchDirections
from .project import project
from .refine import refine
from .gather import gather
from .cluster import Cluster
from .cluster.filter import filter_velocity

import argparse
import sys
from pathlib import Path
import numpy as np
import astropy.units as u
import logging
from time import time

logging.basicConfig()
log = logging.getLogger(__name__)

def main():
    t1 = time()
    log.info("start %s", __name__)

    parser = argparse.ArgumentParser(prog=__name__)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--dx", required=True, type=float)
    parser.add_argument("--velocity", required=True, nargs=2, type=float)
    parser.add_argument("--angle", required=True, nargs=2, type=float)
    parser.add_argument("--threshold", required=True, type=int)
    parser.add_argument("--threshold-type", required=True, type=str)
    parser.add_argument("--output-clusters", type=Path, required=True)
    parser.add_argument("--output-gathered", type=Path, required=True)
    parser.add_argument("--no-refine", action='store_true')

    args = parser.parse_args()
    
    log.info("start loading image PSFs %s", time() - t1)
    images = read(args.images)
    psfs = []
    for image in images:
        bbox = image.reader.readBBox()
        psf = image.reader.readPsf()
        psfs.append(psf.computeShape(bbox.getCenter()).getDeterminantRadius())
    
    log.info("end loading image PSFs %s", time() - t1)
    pixel_scale = (0.263 * u.arcsec / u.pixel)
    psfs = np.array(psfs) * u.pixel * pixel_scale
    log.info("seeing: [%f, %f, %f]", np.min(psfs).value, np.median(psfs).value, np.max(psfs).value)

    dx = args.dx * np.median(psfs)
    log.info(f"using dx = {dx}")

    log.info("start reading catalog %s", time() - t1)
    catalog = read(args.catalog)
    log.info("end reading catalog %s", time() - t1)

    X = catalog.X()
    reference_epoch = X[:, 2].min()
    dt = (X[:, 2].max() - X[:, 2].min())*u.day
    vmin = args.velocity[0] * u.deg/u.day
    vmax = args.velocity[1] * u.deg/u.day
    phimin = args.angle[0] * u.deg
    phimax = args.angle[1] * u.deg

    log.info("start making directions %s", time() - t1)
    directions = SearchDirections([vmin, vmax], [phimin, phimax], dx, dt)
    log.info("end making directions %s", time() - t1)


    log.info("start projecting %s", time() - t1)
    projection = project(X, directions, reference_epoch)
    log.info("end projecting %s", time() - t1)

    log.info("start cluster %s", time() - t1)
    hough = Hough(projection, dx, dx)
    log.info("end cluster %s", time() - t1)

    log.info("start find clusters %s", time() - t1)
    clusters = {}
    for i, cluster in enumerate(hough):
        votes = cluster.extra['votes']
        if args.threshold_type == 'votes' and votes < args.threshold:
            break
        if args.threshold_type == 'clusters' and len(clusters) > args.threshold:
            break
        clusters[i] = cluster
    log.info("end find clusters %s", time() - t1)

    args.output_clusters.parent.mkdir(exist_ok=True, parents=True)
    write(clusters, args.output_clusters)

    if not args.no_refine:
        # refine
        log.info("start refine %s", time() - t1)
        refined = {}
        for i, cluster in clusters.items():
            line = refine(cluster)
            if line:
                refined[i] = line
        log.info("end refine %s", time() - t1)

        # gather
        log.info("start gather %s", time() - t1)
        gathered = {}
        for i, line in refined.items():
            mask = gather(line['line'], X[:,0]*u.deg, X[:,1]*u.deg, X[:, 2]*u.day, dx)
            gathered[i] = Cluster(
                points=X[mask],
                line=line['line'],
            )
        log.info("end gather %s", time() - t1)

        args.output_gathered.parent.mkdir(exist_ok=True, parents=True)
        write(gathered, args.output_gathered)

if __name__ == "__main__":
    main()
