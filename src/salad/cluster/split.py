def join(cluster, catalog):
    import astropy.table
    x = catalog.X(columns=["i_x", "i_y", "ra", "dec", 'exposures', "time", "peakValue", "significance"])
    x = astropy.table.Table(
        x,
        names=['i_x', 'i_y', 'ra', 'dec', 'exposures', 'time', 'peakValue', 'significance']
    )
    p = astropy.table.join(
        x,
        astropy.table.Table(
            cluster.points,
            names=['ra', 'dec', 'time']
        ),
    )
    return p

def main():
    import argparse
    import sys
    import os
    from ..serialize import read, write

    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs='?', type=argparse.FileType('rb'), default=sys.stdin)
    parser.add_argument('--output-format', type=str)

    args = parser.parse_args()
    clusters = read(args.input)
    output_format = args.output_format
    for i, cluster in enumerate(clusters):
        os.makedirs(os.path.dirname(output_format % i), exist_ok=True)
        write(cluster, output_format % i)


if __name__ == "__main__":
    main()