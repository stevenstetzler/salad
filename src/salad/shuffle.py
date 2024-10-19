from .serialize import read, write
from .catalog import DetectionCatalog, MultiEpochDetectionCatalog

import numpy as np
import sys

def shuffle(catalog):
    times = {}
    for c in catalog.single_epoch_catalogs:
        times[c.exposure] = c._time
        
    new_catalogs = []

    exposures = catalog.exposure
    unique_exposures = np.unique(exposures)
    shuffled = np.random.permutation(unique_exposures)
    for orig, new in zip(unique_exposures, shuffled):
        c = list(filter(lambda c : c.exposure == orig, catalog.single_epoch_catalogs))[0]
        new_catalogs.append(DetectionCatalog(c.catalog, times[new], new, c.detector, c.masked_pixel_summary))

    new_catalog = MultiEpochDetectionCatalog(new_catalogs)
    return new_catalog

def main():
    import argparse
    parser = argparse.ArgumentParser(prog=__name__)
    parser.add_argument("input", nargs="?", type=argparse.FileType('rb'), default=sys.stdin)
    parser.add_argument('output', nargs='?', type=argparse.FileType('wb'), default=sys.stdout)
    args = parser.parse_args()

    write(shuffle(read(args.input)), args.output)


if __name__ == "__main__":
    main()
