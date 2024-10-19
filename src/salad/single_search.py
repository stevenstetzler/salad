from .directions import SearchDirections
from .hough import Hough
from .project import project
from .analysis.summary import summary
from .cluster import Cluster
from .gather import gather
from .refine import refine
from .cluster.filter import filter_n, filter_velocity
from .serialize import read, write
from .fakes.recover import recover
from .cluster.join import join
from .cutouts.get import attach_cutouts_from_exposures
import numpy as np
import astropy.units as u
import logging
import os
import joblib

logging.basicConfig()
log = logging.getLogger(__name__)

def single_search(args):
    # fakes
    fakes = read(args.fakes)
    # images
    images = read(args.images)

    # catalog
    catalog = read(args.input)
    X = catalog.X(columns=['ra', 'dec', 'time', 'exposures'])
    ra, dec, time, expnum = X[:, 0] * u.deg, X[:, 1] * u.deg, X[:, 2] * u.day, X[:, 3].astype(int)

    # directions
    dt = (max(catalog.time) - min(catalog.time))*u.day
    reference_time = min(catalog.time) * u.day
    v_range, phi_range, dx = (min(args.velocity)*u.deg/u.day, max(args.velocity)*u.deg/u.day), (min(args.angle)*u.deg, max(args.angle)*u.deg), args.dx*u.arcsec

    b = SearchDirections(v_range, phi_range, dx, dt)

    # projection
    projection = project(X, b, reference_time)

    # cluster
    hough = Hough(projection, dx, dx)
    hough.max()

    # get clusters
    clusters = []
    for cluster in hough:
        votes = cluster.extra['votes']
        if votes < args.vote_threshold:
            break
        clusters.append(cluster)   

    log.info("there are %d results after clustering", len(clusters))
    recovered_clusters = recover(
        fakes, clusters, catalog, match_threshold_line=args.dx/2/3600, projection=projection, hough=hough
    )

    # cluster
    # refine
    # gather + refine
    # gather + refine
    # gather + refine

    # refine iteration 1
    step1_removed = []
    step1 = []
    for i, cluster in enumerate(clusters):
        log.info("refining cluster %d / %d", i, len(clusters))
        result = refine(cluster)
        if result is None:
            step1_removed.append(cluster)
            continue

        mask = gather(result['line'], ra, dec, time, args.gather_threshold[0] * u.arcsec)
        gathered = Cluster(
            points=X[mask], 
            line=result['line'],
            extra=dict(result=result['result']),
        )
        step1.append(gathered)

    log.info("there are %d results after refinement", len(step1))    
    recovered_1 = recover(fakes, step1, catalog)
    recovered_1_removed = recover(fakes, step1_removed, catalog)    

    # refine iteration 2
    step2_removed = []
    step2 = []
    for i, cluster in enumerate(step1):
        log.info("refining cluster %d / %d", i, len(step1))
        result = refine(cluster)
        if result is None:
            step2_removed.append(cluster)
            continue

        mask = gather(result['line'], ra, dec, time, args.gather_threshold[1] * u.arcsec)
        gathered = Cluster(
            points=X[mask], 
            line=result['line'],
            extra=dict(result=result['result']),
        )
        step2.append(gathered)

    log.info("there are %d results after refinement", len(step2))
    recovered_2 = recover(fakes, step2, catalog)
    recovered_2_removed = recover(fakes, step2_removed, catalog)

    # filter velocity and number of detections
    filtered = []
    filtered_removed_velocity = []
    filtered_removed_points = []
    for cluster in step2:
        if filter_velocity(cluster, vmin=min(args.velocity), vmax=max(args.velocity)):
            if filter_n(cluster, n=args.min_points):
                filtered.append(cluster)
                continue
            else:
                filtered_removed_points.append(cluster)
        else:
            filtered_removed_velocity.append(cluster)

    log.info("there are %d results after filtering", len(filtered))

    recovered_filter = recover(fakes, filtered, catalog)
    recovered_filter_removed_velocity = recover(fakes, filtered_removed_velocity, catalog)
    recovered_filter_removed_points = recover(fakes, filtered_removed_points, catalog)

    # load exposures
    exposures = {}
    def get_image(image):
        image.exposure

    log.info("loading exposures")
    joblib.Parallel(n_jobs=args.processes, prefer='threads')(
        joblib.delayed(get_image)(image)
        for image in images
    )
    for i, image in enumerate(images):
        log.info("loading exposure %d/%s", i, len(images))
        exposures[image.expnum] = image.exposure
    exp_times = np.array(list(map(lambda x : x.mjd_mid, images)))*u.day

    # # get cutouts
    log.info("attaching cutouts to clusters")
    for c in [clusters, step1_removed, step1, step2_removed, step2, filtered, filtered_removed_velocity, filtered_removed_points]:
        attach_cutouts_from_exposures(c, exposures)
        for x in c:
            x.summary = summary(x)

    # fit lines on images


    outputs = [
        "hough", 
        "clusters",
        "recovered_clusters",
        "step1_removed", "step1", 
        "recovered_1", "recovered_1_removed",
        "step2_removed", "step2", 
        "recovered_2", "recovered_2_removed",
        "filtered", "filtered_removed_velocity", "filtered_removed_points",
        "recovered_filter", "recovered_filter_removed_velocity", "recovered_filter_removed_points",
    ]
    return {
        o: locals()[o] for o in outputs
    }

import argparse
import sys
parser = argparse.ArgumentParser(prog=__name__)
parser.add_argument("input", nargs="?", type=argparse.FileType('rb'), default=sys.stdin)
parser.add_argument('output', type=str)
parser.add_argument("--fakes", required=True)
parser.add_argument("--images", required=True)
parser.add_argument("--velocity", nargs=2, type=float, default=[0.1, 0.5])
parser.add_argument("--angle", nargs=2, type=float, default=[120, 240])
parser.add_argument("--dx", type=float, default=10)
parser.add_argument("--vote-threshold", type=int, default=25)
parser.add_argument("--min-points", type=int, default=15)
parser.add_argument("--processes", "-J", type=int, default=1)
parser.add_argument("--gather-threshold", nargs="+", type=float, default=[1, 1, 1])

def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser(prog=__name__)
    parser.add_argument("input", nargs="?", type=argparse.FileType('rb'), default=sys.stdin)
    parser.add_argument('output', type=str)
    parser.add_argument("--fakes", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--velocity", nargs=2, type=float, default=[0.1, 0.5])
    parser.add_argument("--angle", nargs=2, type=float, default=[120, 240])
    parser.add_argument("--dx", type=float, default=10)
    parser.add_argument("--vote-threshold", type=int, default=25)
    parser.add_argument("--min-points", type=int, default=15)
    parser.add_argument("--processes", "-J", type=int, default=1)
    parser.add_argument("--gather-threshold", nargs="+", type=float, default=[1, 1, 1])

    args = parser.parse_args()

    outputs = single_search(args)
    outputs['args'] = dict(**vars(args))
    if args.input is sys.stdin:
        outputs['args'].pop("input")
    
    os.makedirs(args.output, exist_ok=True)
    for o in outputs:
        p = os.path.join(args.output, f"{o}.pkl")
        log.info("writing to %s", p)
        write(outputs[o], p)

if __name__ == "__main__":
    main()
