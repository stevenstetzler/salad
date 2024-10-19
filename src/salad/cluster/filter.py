from salad.regression import regression
    
def filter_velocity(cluster, vmin=0.1, vmax=0.5):
    if 'line' in cluster.extra:
        regression_result = cluster.extra['line']
        beta = regression_result.beta.value
    elif hasattr(cluster, 'line'):
        regression_result = cluster.line
        beta = regression_result.beta.value
    else:
        p = cluster.points
        x, y = p[:, 2][:, None], p[:, :2]
        regression_result = regression(x, y)
        if regression_result is None:
            return False
        beta = regression_result.beta
    v = (beta**2).sum()**0.5
    return (v > vmin) and (v <= vmax)

def filter_n(cluster, n=15):
    return len(cluster.points) >= n

def main():
    import argparse
    import sys
    from ..serialize import read, write
    
    parser = argparse.ArgumentParser()

    parser.add_argument("input", nargs='?', type=argparse.FileType('rb'), default=sys.stdin)
    parser.add_argument("output", nargs='?', type=argparse.FileType('wb'), default=sys.stdout)
    parser.add_argument('--velocity', nargs=2, type=float, default=[])
    parser.add_argument('--angle', nargs=2, type=float, default=[])
    parser.add_argument("--min-points", type=int, default=0)
    
    args = parser.parse_args()

    clusters = read(args.input)
    
    print(f"there are {len(clusters)} clusters before filtering", file=sys.stderr)

    filtered_points = {}
    if args.min_points > 0:
        for i in clusters:
            cluster = clusters[i]
            if filter_n(cluster, args.min_points):
                filtered_points[i] = cluster

    filtered_velocity = {}                
    if args.velocity:
        for i in clusters:
            cluster = clusters[i]
            if filter_velocity(cluster, vmin=min(args.velocity), vmax=max(args.velocity)):
                filtered_velocity[i] = cluster

    filtered = {
        i: clusters[i]
        for i in set(list(filtered_points.keys())).intersection(set(list(filtered_velocity.keys())))
    }

    print(f"there are {len(filtered)} clusters after filtering", file=sys.stderr)
    write(filtered, args.output)

if __name__ == "__main__":
    main()
