def join(cluster, catalog):
    import astropy.table
    x = catalog.X(columns=["i_x", "i_y", "ra", "dec", 'exposures', "time", "peakValue", "significance"])
    x = astropy.table.Table(
        x,
        names=['i_x', 'i_y', 'ra', 'dec', 'expnum', 'time', 'peakValue', 'significance']
    )
    p = astropy.table.join(
        x,
        astropy.table.Table(
            cluster.points,
            names=['ra', 'dec', 'time', 'expnum']
        ),
    )
    return p


def main():
    import argparse
    import sys
    from ..serialize import read, write

    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs='?', type=argparse.FileType('rb'), default=sys.stdin)
    parser.add_argument("output", nargs='?', type=argparse.FileType('wb'), default=sys.stdout)
    parser.add_argument('--catalog', type=str, required=True)
    
    args = parser.parse_args()

    cluster = read(args.input)
    catalog = read(args.catalog)

    if isinstance(cluster, list):
        clusters = cluster
        for cluster in clusters:
            cluster.extra['join'] = join(cluster, catalog)
        out = clusters
    else:
        cluster.extra['join'] = join(cluster, catalog)
        out = cluster

    write(out, args.output)

if __name__ == "__main__":
    main()
