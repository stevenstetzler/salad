from ..io.serialize import read, write
import logging
from .forced import forced_exposures

logging.basicConfig()
log = logging.getLogger(__name__)

# from a catalog and clusters
# measure the lines

def forced_points(cluster):
    import astropy.table
    cutouts = cluster.extra['cutouts']
    points = cluster.extra['join']
    photometry = forced_exposures(cutouts, points)
    # p['forced_flux'] = fluxes
    # p['forced_mag'] = mags
    # p['forced_mag_err_low'] = magerr_low
    # p['forced_mag_err_high'] = magerr_high
    # p['forced_a'] = a
    # p['forced_c'] = c
    return astropy.table.hstack([points, photometry])

def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs='?', type=argparse.FileType('rb'), default=sys.stdin)
    parser.add_argument('output', nargs='?', type=argparse.FileType('wb'), default=sys.stdout)

    args = parser.parse_args()

    cluster = read(args.input)
    measurement = forced_points(cluster)
    cluster.extra['forced_points'] = measurement
    cluster.extra.pop("cutouts")
    write(cluster, args.output)

if __name__ == "__main__":
    main()
