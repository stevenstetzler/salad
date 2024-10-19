import importlib
import argparse
import logging
import sys

logging.basicConfig()
log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("subcommand", type=str)
    parser.add_argument("--log-level", type=str, default="INFO")

    args, _ = parser.parse_known_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    sys.argv = [sys.argv[0]] + sys.argv[sys.argv.index(args.subcommand) + 1:]

    module = importlib.import_module(f".{args.subcommand}", "salad")
    if hasattr(module, "log"):
        module.log.setLevel(getattr(logging, args.log_level.upper()))
    if hasattr(module, "main"):
        module.main()
    else:
        raise RuntimeError(f"main is not defined for module salad.{args.subcommand}")

if __name__ == "__main__":
    main()
