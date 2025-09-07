import numpy as np

def filter_fakes(fakes, refs):
    def get_visit_numbers(refs):
        return list(set([ref.dataId['visit'] for ref in refs]))
    
    def get_detector_numbers(refs):
        return list(set([ref.dataId['detector'] for ref in refs]))

    visits = get_visit_numbers(refs)    
    detectors = get_detector_numbers(refs)

    mask = np.any([fakes['EXPNUM'] == v for v in visits], axis=0)
    mask &= np.any([fakes['CCDNUM'] == d for d in detectors], axis=0)
    return fakes[mask]
    
def main():
    import argparse
    import lsst.daf.butler as dafButler
    import pickle
    import sys    

    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=argparse.FileType('rb'), default=sys.stdin)
    parser.add_argument('output', nargs='?', type=argparse.FileType('wb'), default=sys.stdout)
    parser.add_argument("--repo", type=str)
    parser.add_argument("--datasetType", type=str)
    parser.add_argument("--collections", type=str, default="*")
    parser.add_argument("--where", type=str, default=None)

    args = parser.parse_args()

    if args.input == sys.stdin:
        args.input = sys.stdin.buffer
    fakes = pickle.load(args.input)

    butler = dafButler.Butler(args.repo)
    refs = list(butler.registry.queryDatasets(
        args.datasetType, 
        collections=butler.registry.queryCollections(args.collections), 
        where=args.where
    ))
    fakes = filter_fakes(fakes, refs)
    print("there are", len(fakes), "fakes after filtering", file=sys.stderr)

    if args.output is sys.stdout:
        args.output = sys.stdout.buffer
    pickle.dump(fakes, args.output)
