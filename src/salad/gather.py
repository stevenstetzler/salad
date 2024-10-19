import numpy as np
import sys
import logging
import astropy.units as u
from .serialize import read, write
from .cluster.cluster import Cluster
from .regression import RegressionResult
from .detection import MultiEpochDetectionCatalog

logging.basicConfig()
log = logging.getLogger(__name__)

def gather(result : RegressionResult, ra, dec, time, threshold):
    """
    gather all points within some threshold of a line
    """
    sky_units = ra.unit
    time_units = time.unit
    y_pred = np.dot(time[:, None], result.beta) + result.alpha
    # y_pred = ((np.array([time]).T * time_units @ result.beta) * sky_units + result.alpha)
    residuals = (
        np.array([ra, dec]).T * sky_units - 
        y_pred
    )
    distance = (residuals**2).sum(axis=1)**0.5
    mask = distance < threshold
    log.info("gathered %s points", mask.sum())
    return mask

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=argparse.FileType('rb'), default=sys.stdin)
    parser.add_argument('output', nargs='?', type=argparse.FileType('wb'), default=sys.stdout)
    parser.add_argument('--catalog', type=str, required=True)

    parser.add_argument("--ra-column", type=str, default="ra")
    parser.add_argument("--dec-column", type=str, default="dec")
    parser.add_argument("--time-column", type=str, default="time")

    parser.add_argument("--sky-units", default="deg", type=str)
    parser.add_argument("--time-units", default="day", type=str)

    parser.add_argument("--threshold", type=float, default=1)
    parser.add_argument("--threshold-units", default="arcsec", type=str)

    args = parser.parse_args()

    threshold = args.threshold * getattr(u, args.threshold_units)
    catalog = read(args.catalog)
    lines = read(args.input)

    sky_units = getattr(u, args.sky_units)
    time_units = getattr(u, args.time_units)
    X = catalog.X(columns=[args.ra_column, args.dec_column, args.time_column, "exposures"], sky_units=sky_units, time_units=time_units)
    ra = X[:, 0] * sky_units
    dec = X[:, 1] * sky_units
    time = X[:, 2] * time_units
    
    clusters = {}
    for i in lines:
        if lines[i] is not None:
            result = lines[i]['result']
            if result is not None:
                result.beta *= sky_units / time_units
                result.alpha *= sky_units
                mask = gather(result, ra, dec, time, threshold)
                clusters[i] = Cluster(
                    points=X[mask], 
                    line=lines[i]['line'], 
                    extra=dict(result=result),
                )
        else:
            log.warn("skipping line with no regression result")
    
    write(clusters, args.output)

if __name__ == "__main__":
    main()
