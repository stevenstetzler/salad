import logging
import astropy.units as u
import pickle
from .catalog import MultiEpochDetectionCatalog
from .primitives import transform_to_xy_prime
from .directions import SearchDirections
from .serialize import Serializable

logging.basicConfig()
log = logging.getLogger(__name__)

class Projection(Serializable):
    directions = None
    X = None
    projected = None
    reference_time = None

def project(X, directions, reference_time):
    """
    Projects points using provided guess velocities
    """
    projection = Projection()
    projection.X = X
    projection.reference_time = reference_time
    projection.directions = directions

    log.info(f"projecting {len(X)} points using {len(directions.b)} directions")
    projection.projected = transform_to_xy_prime(X, directions.b.value, reference_time)
    return projection

def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser(prog=__name__)
    parser.add_argument("input", nargs="?", type=argparse.FileType('rb'), default=sys.stdin)
    parser.add_argument('output', nargs='?', type=argparse.FileType('wb'), default=sys.stdout)
    parser.add_argument("--velocity", nargs=2, type=float, required=True)
    parser.add_argument("--angle", nargs=2, type=float, required=True)
    parser.add_argument("--dx", type=float, required=True)
    parser.add_argument("--ra-column", type=str, default="ra")
    parser.add_argument("--dec-column", type=str, default="dec")
    parser.add_argument("--time-column", type=str, default="time")
    parser.add_argument("--sky-units", default="deg", type=str)
    parser.add_argument("--time-units", default="day", type=str)
    parser.add_argument("--angle-units", default="deg", type=str)
    parser.add_argument("--reference-time", default=None, type=float)
    parser.add_argument("--dx-units", default="arcsec", type=str)
    
    args = parser.parse_args()

    sky_units = getattr(u, args.sky_units)
    time_units = getattr(u, args.time_units)
    angle_units = getattr(u, args.angle_units)
    dx_units = getattr(u, args.dx_units)

    catalog = MultiEpochDetectionCatalog.read(args.input)
    v = [
        args.velocity[0] * sky_units / time_units,
        args.velocity[1] * sky_units / time_units,
    ]
    phi = [
        args.angle[0] * angle_units,
        args.angle[1] * angle_units,
    ]
    dx = args.dx * dx_units
    X = catalog.X(columns=[args.ra_column, args.dec_column, args.time_column, "exposures"], sky_units=sky_units, time_units=time_units)
    dt = (X[:, 2].max() - X[:, 2].min()) * time_units
    directions = SearchDirections(
        v, phi,
        dx, dt,
    )
    print(directions.b, file=sys.stderr)
    if args.reference_time is None:
        args.reference_time = catalog.X(columns=[args.time_column]).min()
    
    projection = project(X, directions, args.reference_time)
    if args.output is sys.stdout:
        args.output = sys.stdout.buffer
    
    pickle.dump(projection, args.output)

if __name__ == "__main__":
    main()
