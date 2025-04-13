from ..serialize import read
import sys

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, default=sys.stdin)
    parser.add_argument('output', type=str, default=sys.stdout)
    parser.add_argument("--format", type=str, default="ascii.ecsv")

    args = parser.parse_args()

    read(args.input).to_table().write(args.output, format=args.format, overwrite=True)

if __name__ == "__main__":
    main()
