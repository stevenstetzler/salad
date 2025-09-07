from ..io.serialize import read, write
import logging

logging.basicConfig()
log = logging.getLogger(__name__)

def get_positions(cluster, times):
    line = cluster.extra['line']
    positions = (times[:, None] @ line.beta) + line.alpha
    return positions

def coadd(cutouts, coadd_type="nanmean"):
    import numpy as np
    return getattr(np, coadd_type)(cutouts, axis=0)

def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs='?', type=argparse.FileType('rb'), default=sys.stdin)
    parser.add_argument('output', nargs='?', type=argparse.FileType('wb'), default=sys.stdout)

    args = parser.parse_args()

    cluster = read(args.input)
    cutouts = [cutout.image.array for cutout in cluster.extra['cutouts'] if cutout.image.array.shape == (50, 50)]
    coadd_array = coadd(cutouts)
    write(coadd_array, args.output)    

if __name__ == "__main__":
    main()
