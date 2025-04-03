from .serialize import read, write
from .hough import Hough, LikelihoodHough
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

logging.basicConfig()
log = logging.getLogger(__name__)

def main():
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
    parser.add_argument("--sum-column", type=str, default="count")
    parser.add_argument("--per-direction", action="store_true")
    parser.add_argument("--filter-likelihood", type=float, default=-1.)

    args = parser.parse_args()
    
    images = read(args.images)
    psfs = []
    for image in images:
        bbox = image.reader.readBBox()
        psf = image.reader.readPsf()
        psfs.append(psf.computeShape(bbox.getCenter()).getDeterminantRadius())

    pixel_scale = (0.263 * u.arcsec / u.pixel)
    psfs = np.array(psfs) * u.pixel * pixel_scale
    log.info("seeing: [%f, %f, %f]", np.min(psfs).value, np.median(psfs).value, np.max(psfs).value)

    dx = args.dx * np.median(psfs)
    log.info(f"using dx = {dx}")
    catalog = read(args.catalog)

    columns = ['ra','dec','time','exposures']
    if (args.filter_likelihood > -1) or (args.sum_column == 'likelihood'):
        columns.extend(['psi', 'phi'])
    elif args.sum_column != 'count':
        columns.append(args.sum_column)
    
    X = catalog.X(columns=columns)
    reference_epoch = X[:, 2].min()
    dt = (X[:, 2].max() - X[:, 2].min())*u.day
    vmin = args.velocity[0] * u.deg/u.day
    vmax = args.velocity[1] * u.deg/u.day
    phimin = args.angle[0] * u.deg
    phimax = args.angle[1] * u.deg

    def make_hough(projection):
        if args.sum_column == 'likelihood':
            hough = LikelihoodHough(projection, dx, dx, values=[columns.index("psi"), columns.index("phi")])
        else:
            if args.sum_column == 'count':
                hough = Hough(projection, dx, dx)
            else:
                if isinstance(args.sum_column, str):
                    i = columns.index(args.sum_column)
                elif isinstance(args.sum_column, int):
                    i = args.sum_column
                else:
                    raise RuntimeError(f"sum column {args.sum_column} not supported")
                hough = Hough(projection, dx, dx, values=i, dtype=np.float64)
        return hough

    def find_clusters(hough):
        clusters = {}
        for i, cluster in enumerate(hough):
            votes = cluster.extra['votes']
            if args.filter_likelihood > -1:
                nu = (cluster.points[:, columns.index("psi")] / cluster.points[:, columns.index("phi")]**0.5).sum()
                log.info("nu_coadd = %s", nu)
                if nu < args.filter_likelihood:
                    log.info("removing candidate since %s < %s", nu, args.filter_likelihood)
                    continue
            if args.threshold_type == 'votes' and votes < args.threshold:
                break
            if args.threshold_type == 'clusters' and len(clusters) > args.threshold:
                break
            clusters[i] = cluster
        return clusters

    directions = SearchDirections([vmin, vmax], [phimin, phimax], dx, dt)

    if args.per_direction:
        clusters = {}
        i = 0
        for direction in directions.b:
            projection = project(
                X,
                SearchDirections(None, None, dx, dt, b=direction[None, :]),
                reference_epoch
            )
            direction_clusters = find_clusters(make_hough(projection))
            c = 0
            for k, cluster in direction_clusters.items():
                clusters[i + k] = cluster
                c = max(c, k)
            i = c + 1
    else:
        projection = project(X, directions, reference_epoch)
        clusters = find_clusters(make_hough(projection))

    # refine
    refined = {}
    for i, cluster in clusters.items():
        line = refine(cluster)
        if line:
            refined[i] = line

    # gather
    gathered = {}
    for i, line in refined.items():
        mask = gather(line['line'], X[:,0]*u.deg, X[:,1]*u.deg, X[:, 2]*u.day, dx)
        gathered[i] = Cluster(
            points=X[mask],
            line=line['line'],
        )

    args.output_clusters.parent.mkdir(exist_ok=True, parents=True)
    write(clusters, args.output_clusters)
    args.output_gathered.parent.mkdir(exist_ok=True, parents=True)
    write(gathered, args.output_gathered)

if __name__ == "__main__":
    main()
