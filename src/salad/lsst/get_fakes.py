def main():
    import argparse
    import lsst.daf.butler as dafButler
    import pickle
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=str)
    parser.add_argument("datasetType", type=str)
    parser.add_argument('--output', nargs='?', type=argparse.FileType('wb'), default=sys.stdout)
    parser.add_argument("--fakesType", type=str, default=None)
    parser.add_argument("--collections", type=str, default="*")
    # parser.add_argument("--where", type=str, default=None)
    # parser.add_argument("--processes", "-J", type=int, default=1)

    args = parser.parse_args()

    butler = dafButler.Butler(args.repo)
    fakes = butler.get(args.datasetType, collections=butler.registry.queryCollections(args.collections)).asAstropy()
    if args.fakesType:
        fakes = fakes[fakes['type'] == args.fakesType]

    print("there are", len(fakes), "fakes", file=sys.stderr)
    
    if args.output is sys.stdout:
        args.output = sys.stdout.buffer
    pickle.dump(fakes, args.output)



if __name__ == "__main__":
    main()
