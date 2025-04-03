import logging
from ..serialize import Serializable, read, write

logging.basicConfig()
log = logging.getLogger(__name__)

class Cluster(object):
    points = None
    line = None
    cutouts = None
    extra = {}
    summary = None
    
    def __init__(self, **kwargs):
        for k in kwargs:
            setattr(self, k, kwargs[k])

# class Clusters(Serializable):
#     def __init__(self, clusters=[]):
#         self.clusters = clusters

#     def add(self, cluster):
#         self.clusters.append(cluster)

#     def __iter__(self):
#         return iter(self.clusters)

def main():
    import argparse
    import sys
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs='?', type=argparse.FileType('rb'), default=sys.stdin)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument("--action", type=str, nargs="+", required=True)

    args = parser.parse_args()

    if args.action[0] == "split":
        clusters = read(args.input)
        os.makedirs(args.output_dir, exist_ok=True)
        for i, cluster in enumerate(clusters):
            write(cluster, os.path.join(args.output_dir, f"cluster_{i}.pkl"))
    elif args.action[0] == "length":
        clusters = read(args.input)
        write(len(clusters), file=args.output)
    else:
        raise Exception(f"action {args.action[0]} is not supported")

if __name__ == "__main__":
    main()