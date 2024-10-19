import numpy as np
import sys
import logging
from .serialize import Serializable, read, write
from .cluster.cluster import Cluster
from .regression import regression, RegressionResult
from .line import Line

logging.basicConfig()
log = logging.getLogger(__name__)

def refine(cluster : Cluster):
    p = cluster.points
    log.info(f"refining cluster with {len(p)} points")
    if len(p) < 2:
        log.warn("cluster has too few points to fit a line")
        return None
    mask = np.ones(len(p)).astype(bool)
    x, y = p[mask, 2][:, None], p[mask, :2]
    try:
        regression_result = regression(x, y)
        if regression_result is None:
            log.warn("regression on points in cluster failed")
            return None
    except Exception as e:
        log.exception(e)
        return None
    regression_error = (np.diag(regression_result.sigma_e)**2).sum()
    log.debug(f"regression error {regression_error}")
    outliers = regression_result.outliers_r
    inliers = ~outliers
    # if outliers.sum() > (len(x) / 4): # there is a chance for confusion between inliers/outliers
    #     log.debug(f"regressing {inliers.sum()} inliers and {outliers.sum()} outliers separately")
    #     inlier_regression = regression(x[inliers], y[inliers])
    #     outlier_regression = regression(x[outliers], y[outliers])
    #     if inlier_regression is None and outlier_regression is None:
    #         log.warning(f"both inlier and outlier regression are invalid; will use original regression result")
    #         result = regression_result
    #     elif inlier_regression is not None and outlier_regression is None:
    #         log.warning(f"outlier regression is invalid; will use inlier regression result")
    #         result = inlier_regression
    #     elif inlier_regression is None and outlier_regression is not None:
    #         log.warning(f"inlier regression is invalid; will use outlier regression result")
    #         result = outlier_regression
    #     elif inlier_regression is not None and outlier_regression is not None:
    #         inlier_error = (np.diag(inlier_regression.sigma_e)**2).sum()
    #         outlier_error = (np.diag(outlier_regression.sigma_e)**2).sum()
    #         log.debug(f"inlier error {inlier_error}, outlier_error {outlier_error}")
    
    #         if regression_error > inlier_error and regression_error > outlier_error:
    #             if inlier_error > outlier_error:
    #                 result = outlier_regression
    #             else:
    #                 result = inlier_regression
    #         elif regression_error > inlier_error:
    #             result = inlier_regression
    #         elif regression_error > outlier_error:
    #             result = outlier_regression
    #         else:
    #             result = regression_result
    #     result_error = (np.diag(result.sigma_e)**2).sum()
    #     log.debug(f"final error {result_error}")
    # else:
    #     result = regression_result
    result = regression_result
    outliers = result.outliers_r
    inliers = ~outliers
    log.info(f"refined cluster has {inliers.sum()} inliers")

    line = Line(
        alpha=result.alpha * cluster.line.alpha.unit,
        beta=result.beta * cluster.line.beta.unit
    )
    return dict(
        result=result,
        line=line
    )

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=argparse.FileType('rb'), default=sys.stdin)
    parser.add_argument('output', nargs='?', type=argparse.FileType('wb'), default=sys.stdout)

    args = parser.parse_args()
    clusters = read(args.input)
    refined = {i: refine(cluster) for i, cluster in clusters.items()}
    write(refined, args.output)

if __name__ == "__main__":
    main()
