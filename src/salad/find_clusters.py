import logging
from .hough import Hough
from .serialize import read, write
from .cluster.cluster import Cluster

logging.basicConfig()
log = logging.getLogger(__name__)

def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser(prog=__name__)
    parser.add_argument("input", nargs="?", type=argparse.FileType('rb'), default=sys.stdin)
    parser.add_argument('output', nargs='?', type=argparse.FileType('wb'), default=sys.stdout)
    parser.add_argument("--threshold", type=int, required=True)

    args = parser.parse_args()

    hough = read(args.input)
    
    clusters = {}
    # This is what it looks like to find all clusters in a hough space
    # points = hough.projection.X.copy()
    for i, cluster in enumerate(hough):
        votes = cluster.extra['votes']
        if votes < args.threshold:
            break
        clusters[i] = cluster
    write(clusters, args.output)

if __name__ == "__main__":
    main()
