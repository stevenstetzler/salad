import numpy as np
import logging
import astropy.units as u
from .primitives import digitize_xy, vote_bins, close_to_line
from .project import Projection
from .serialize import Serializable
from .cluster.cluster import Cluster
from .line import Line

logging.basicConfig()
log = logging.getLogger(__name__)

class Hough(Serializable):
    def __init__(self, projection, dx, dy, dtype=np.int32):
        self.projection = projection
        self.mask = np.ones(len(projection.X)).astype(bool)
        # these are the original sky units of the catalog
        self.sky_units = (self.projection.directions.v_min.unit * self.projection.directions.dt.unit)
        self.time_units = self.sky_units / self.projection.directions.v_min.unit
        self.dx = dx.to(self.sky_units).value
        self.dy = dy.to(self.sky_units).value

        self.min_x = self.projection.projected[:, :, 0].min()
        self.max_x = self.projection.projected[:, :, 0].max()
        self.min_y = self.projection.projected[:, :, 1].min()
        self.max_y = self.projection.projected[:, :, 1].max()
        
        self.shape = (
            self.projection.projected.shape[0], 
            int(((self.max_x - self.min_x) / self.dx) + 1),
            int(((self.max_y - self.min_y) / self.dy) + 1),
        )
        log.info("creating hough space with shape %s", self.shape)
        self.array = np.zeros(self.shape, dtype=dtype)
        log.info("digitizing projected catalog")
        self.bins = digitize_xy(self.projection.projected, self.min_x, self.min_y, self.dx, self.dy)
        log.info("voting")
        self.vote()
        log.info("max vote %s at %s", self.max(), self.argmax())

    def argmax(self):
        return np.unravel_index(self.array.argmax(), self.array.shape)
    
    def max(self):
        return self.array.max()
    
    def vote(self, mask=None, **kwargs):
        if mask is not None:
            bins = self.bins[:, mask]
        else:
            bins = self.bins

        return vote_bins(self.array, bins, **kwargs)
    
    def __iter__(self):
        return self

    def __next__(self):
        b, x, y = self.argmax()
        votes = self.array[b, x, y]
        log.info("next cluster has %d votes", votes)
        mask = (self.bins[b, :, 0] == x) & (self.bins[b, :, 1] == y)
        points = self.projection.X[mask]
        self.vote(mask=mask, coef=-1)
        self.mask &= ~mask
        log.info("%d points remain", self.mask.sum())
        line = Line(
            alpha=np.array(
                [
                    (x * self.dx + self.min_x + self.dx/2), 
                    (y * self.dy + self.min_y + self.dy/2)
                ]
            ) * self.sky_units,
            beta=self.projection.directions.b[b],
            offset=self.projection.reference_time
        )
        
        return Cluster(points=points, line=line, extra=dict(votes=votes, b=b, y=y, x=x))

    # def vote(self, *args, mask=None, use_numba=True, **kwargs):

def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser(prog=__name__)
    parser.add_argument("input", nargs="?", type=argparse.FileType('rb'), default=sys.stdin)
    parser.add_argument('output', nargs='?', type=argparse.FileType('wb'), default=sys.stdout)
    parser.add_argument("--dx", type=float, required=True)
    parser.add_argument("--dx-units", default="arcsec", type=str)
    parser.add_argument("--dy", type=float, default=None)
    parser.add_argument("--dy-units", default="arcsec", type=str)

    args = parser.parse_args()
    dx_units = getattr(u, args.dx_units)
    dy_units = getattr(u, args.dy_units)
    dx = args.dx * dx_units
    if args.dy is None:
        args.dy = args.dx
    dy = args.dy * dy_units
    projection = Projection.read(args.input)
    hough = Hough(projection, dx, dy)
    hough.write(args.output)

if __name__ == "__main__":
    main()
