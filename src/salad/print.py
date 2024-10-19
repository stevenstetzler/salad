from salad.serialize import read
import sys
import logging

logging.basicConfig()
log = logging.getLogger(__name__)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=argparse.FileType('rb'), default=sys.stdin)
    parser.add_argument('output', nargs='?', type=argparse.FileType('wb'), default=sys.stdout)
    
    args = parser.parse_args()

    print(read(args.input), file=args.output)

if __name__ == "__main__":
    main()
