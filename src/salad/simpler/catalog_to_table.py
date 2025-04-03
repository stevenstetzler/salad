from ..serialize import read
import sys

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=argparse.FileType('rb'), default=sys.stdin)
    parser.add_argument('output', nargs='?', type=argparse.FileType('wb'), default=sys.stdout)

    args = parser.parse_args()

    read(args.input).to_table().write(args.output, format='ascii.ecsv')

if __name__ == "__main__":
    main()
