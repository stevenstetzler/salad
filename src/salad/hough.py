import numpy as np
import logging
import astropy.units as u
from .primitives import digitize_xy, vote_bins, vote_points, close_to_line
from .project import Projection
from .serialize import Serializable
from .cluster.cluster import Cluster
from .line import Line

logging.basicConfig()
log = logging.getLogger(__name__)

class LikelihoodHough():
    def __init__(self, projection, dx, dy, values):
        self.psi = Hough(projection, dx, dy, values=values[0], dtype=np.float64)
        self.phi = Hough(projection, dx, dy, values=values[1], dtype=np.float64)
        self.update_ratio()

    def update_ratio(self):
        ratio = (self.psi.array / (self.phi.array**0.5))
        ratio[np.isnan(ratio)] = 0
        ratio[~np.isfinite(ratio)] = 0
        self.ratio = ratio

    def argmax(self):
        return np.unravel_index(self.ratio.argmax(), self.ratio.shape)

    def vote(self, mask=None, **kwargs):
        for h in [self.psi, self.phi]:
            if mask is not None:
                bins = h.bins[:, mask]
                values = h.values[mask]
            else:
                bins = h.bins
                values = h.values

            vote_bins(h.array, bins, values, **kwargs)
        self.update_ratio()
    
    def __iter__(self):
        return self

    def __next__(self):
        h = self.psi
        b, x, y = self.argmax()
        nu = self.ratio[b, x, y]
        log.info("next cluster has nu %f", nu)
        in_bin = (h.bins[b, :, 0] == x) & (h.bins[b, :, 1] == y)
        mask = in_bin & h.mask # exclude previously considered points
        self.vote(mask=mask, coef=-1)
        points = h.projection.X[mask]

        self.psi.mask &= ~in_bin # exclude these points from future consideration
        self.phi.mask &= ~in_bin # exclude these points from future consideration
        
        center = (
            x * h.dx + h.min_x + h.dx/2, 
            y * h.dy + h.min_y + h.dy/2
        )
        
        line = Line(
            alpha=np.array([center[0], center[1]]) * h.sky_units,
            beta=h.projection.directions.b[b],
            offset=h.projection.reference_time
        )
        
        return Cluster(
            points=points, 
            line=line, 
            extra=dict(
                votes=nu, 
                b=b, y=y, x=x, 
            )
        )


class Hough(Serializable):
    def __init__(self, projection, dx, dy, values='count', dtype=np.int32):
        self.projection = projection
        n_points = len(projection.X)
        self.mask = np.ones(n_points).astype(bool)
        if isinstance(values, str) and values == "count":
            self.values = np.ones(n_points).astype(dtype)
        elif isinstance(values, int): # column of data points
            self.values = projection.X[:, values]
            dtype = self.values.dtype
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
            points = self.projection.X[mask]
            values = self.values[mask]
        else:
            bins = self.bins
            points = self.projection.X
            values = self.values

        # return vote_bins(self.array, bins, values, **kwargs)
        return vote_points(
            self.array, points, self.projection.directions.b, 
            self.min_x, self.min_y, self.dx, self.dy, self.projection.reference_time, 
            **kwargs
        )
    
    def __iter__(self):
        return self

    def __next__(self):
        b, x, y = self.argmax()
        votes = self.array[b, x, y]
        log.info("next cluster has value %s at %s", votes, (b, x, y))
        in_bin = (self.bins[b, :, 0] == x) & (self.bins[b, :, 1] == y)
        mask = in_bin & self.mask # exclude previously considered points
        points = self.projection.X[mask]
        self.vote(mask=mask, coef=-1)
        self.mask &= ~in_bin # exclude these points from future consideration

        group_mask = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                x_i = x + i
                y_j = y + j
                in_bounds = (
                    0 <= x_i < self.shape[1]
                ) and (
                    0 <= y_j < self.shape[1]
                )
                if in_bounds:
                    group_mask.append((self.bins[b, :, 0] == x_i) & (self.bins[b, :, 1] == y_j))
        
        group_mask = np.logical_or.reduce(group_mask)
        group = self.projection.projected[b, group_mask]
        center = (
            x * self.dx + self.min_x + self.dx/2, 
            y * self.dy + self.min_y + self.dy/2
        )
        in_range = (
            np.abs(group[:, 0] - center[0]) < self.dx
        ) & (
            np.abs(group[:, 1] - center[1]) < self.dy
        )
        group_points = self.projection.X[group_mask][in_range]

        log.info("%d points remain", self.mask.sum())
        line = Line(
            alpha=np.array([center[0], center[1]]) * self.sky_units,
            beta=self.projection.directions.b[b],
            offset=self.projection.reference_time
        )
        
        return Cluster(
            points=points, 
            line=line, 
            extra=dict(
                votes=votes, 
                b=b, y=y, x=x, 
                group=group, 
                in_range=group[in_range], 
                group_points=group_points
            )
        )

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
