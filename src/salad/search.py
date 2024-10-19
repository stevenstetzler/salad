import astropy.units as u
import astropy.table
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import logging
from .catalog import MultiEpochDetectionCatalog
from .primitives import *
from .hough import Hough
from .regression import *
from .serialize import Serializable

logging.basicConfig()
logger = logging.getLogger(__name__)

class SearchResult():
    votes = 0
    x_idx = None
    y_idx = None
    b_idx = None
    hough_anchor = None
    hough_dir = None
    voters = None
    close = None
    refined_anchor = None
    refined_dir = None
    refined = None
    regression_result = None
                
class Search(Serializable):
    def __init__(self, hough):
        self.hough = hough
        self.mask = np.ones(hough.X.shape[0], dtype=bool)
        self.results = []
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)
    
    def refine(self, X, robust=True):
        regression_result = regression(X[:, 2][:, None], X[:, :2], robust=robust)
        if regression_result is not None:
            return np.atleast_1d(self.hough.reference_time) @ regression_result.beta + regression_result.alpha, regression_result.beta[0], regression_result
        else:
            return None, None, None
    
    def get_top_line(self, robust=True, refine_direction=True, refine_anchor=True):
        result = SearchResult()
        result.b_idx, result.x_idx, result.y_idx = self.hough.argmax()
        result.votes = self.hough.max() # self.hough.array[result.b_idx, result.x_idx, result.y_idx]
        self.log.info("top line has %s votes", result.votes)
        result.voters = self.hough.voters(result.b_idx, result.x_idx, result.y_idx) & self.mask
        result.hough_anchor = self.hough.anchor(result.x_idx, result.y_idx)
        result.hough_dir = self.hough.b[result.b_idx]
        # print(hough_anchor, hough_anchor.shape)
        result.close = close_to_line(self.hough.X, result.hough_anchor[None], result.hough_dir, self.hough.tolerance, self.hough.reference_time) & self.mask
        self.log.info("there are %s points close to line %s %s %s", result.close.sum(), close_to_line(self.hough.X, result.hough_anchor[None], result.hough_dir, self.hough.tolerance, self.hough.reference_time).sum(), result.hough_anchor, result.hough_dir)
        if result.close.sum() < 2:
            # cannot refine a result with too few points
            return result
        
        if refine_anchor and not refine_direction:
            # get the (robust) mean of the data points in the x' y' space
            xy = transform_to_xy_prime(self.hough.X[result.close], result.hough_dir[None], self.hough.reference_time)[0]
            _mu, _ = mean_and_covar(xy, robust=robust)
            result.refined_anchor = _mu[:2]
            result.refined_dir = result.hough_dir.copy()
        elif refine_anchor and refine_direction:
            # get the (robust) slope and (robust) intercept of a regression model
            # the new anchor is the position at the reference time using the (robust) slope and (robust) intercept
            # the new direction is the (robust) slope
            result.refined_anchor, result.refined_dir, result.regression_result = self.refine(self.hough.X[result.close], robust=robust)
            # print(refined_anchor, refined_anchor.shape)
            # if result.refined_anchor is None:
            #     raise Exception("error in refinement")
        elif not refine_anchor and refine_direction:
            # the new direction is the (robust) slope
            # use the old anchor
            # undefined?
            raise Exception("can't refine direction and not the anchor")
        else:
            # do nothing
            result.refined_anchor = result.hough_anchor.copy()
            result.refined_dir = result.hough_dir.copy()
        
        if result.refined_anchor is not None and result.refined_dir is not None:
            result.refined = close_to_line(self.hough.X, result.refined_anchor[None], result.refined_dir[None], self.hough.tolerance, self.hough.reference_time) & self.mask
            self.log.info("there are %s points close to line after refinement", result.refined.sum())

        return result

    def __iter__(self):
        return self
    
    def __next__(self):
        self.log.info("getting next result")
        search_result = self.get_top_line()
        if search_result.refined is not None:
            search_mask = search_result.refined
        else:
            search_mask = search_result.close
        self.log.info("removing %s points from consideration %s %s" % (search_mask.sum(), search_result.votes, search_result.voters.sum()))
        if search_mask.sum() == 0 and search_result.votes != 0:
            raise RuntimeError()
        self.hough.vote(mask=search_mask, coef=-1) # subtract the points found in this result from the hough space
        self.mask &= (~search_mask) # update the mask to exclude what was found in this result
        return search_result
        
    def run(self, criteria):
        self.log.info("starting search with criteria: %s", criteria)
        self.criteria = criteria
        for result in self:
            self.log.info("there are %s points remaining", self.mask.sum())
            if criteria.should_stop(self, result, self.results):
                self.log.info("criteria reached; stopping search.")
                break
            else:
                self.results.append(result)
        return self.results

class Stopper():
    def should_stop(search, result, results):
        raise NotImplementedError()
        
class VoteThreshold(Stopper):
    def __init__(self, threshold):
        self.threshold = threshold
        
    def should_stop(self, search, result, results):
        if result.votes < self.threshold:
            return True
        else:
            return False

def search(catalog, v, phi, dx, threshold, sky_units, time_units, stopper):
    X = catalog.X(columns=["ra", "dec", "time"], sky_units=sky_units, time_units=time_units)
    logger.info(f"there are {len(X)} input points")
    dt = (X[:, 2].max() - X[:, 2].min()) * time_units
    directions = SearchDirections(
        v, phi,
        dx, dt,
    )
    logger.info(f"searching {len(directions.b)} directions")
    hough = Hough(X, directions.b.value, dx.to(sky_units).value, dx.to(sky_units).value, threshold)

    hough.vote()
    search = Search(hough)
    search.run(stopper)
    return search

def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser(prog=__name__)
    parser.add_argument("input", nargs="?", type=argparse.FileType('rb'), default=sys.stdin)
    parser.add_argument('output', nargs='?', type=argparse.FileType('wb'), default=sys.stdout)
    parser.add_argument("--columns", nargs="+", default=None)
    parser.add_argument("--velocity", nargs=2, type=float)
    parser.add_argument("--angle", nargs=2, type=float)
    parser.add_argument("--sky-units", default="deg")
    parser.add_argument("--time-units", default="day")
    parser.add_argument("--angle-units", default="deg")
    parser.add_argument("--dx", nargs=1, type=float)
    parser.add_argument("--dx-units", default="arcsec")
    parser.add_argument("--vote-threshold", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.log_level.upper()))

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
    dx = args.dx[0] * dx_units
    if args.vote_threshold is None:
        args.vote_threshold = int(catalog.num_times / 2 + 1)
        logger.info(f"setting vote threshold to {args.vote_threshold}")
    stopper = VoteThreshold(args.vote_threshold)
    s = search(catalog, v, phi, dx, dx.to(sky_units).value, sky_units, time_units, stopper)
    logger.info(f"found {len(s.results)} results")
    if args.output is sys.stdout:
        args.output = sys.stdout.buffer
    pickle.dump(s, args.output)
    
if __name__ == "__main__":
    main()
