from .directions import SearchDirections
from ..io.serialize import read
import numpy as np
from numba import cuda
import astropy.table
from .postprocess import refine, gather

import astropy.units as u
from pathlib import Path
import logging
from joblib import Parallel, delayed

logging.basicConfig()
log = logging.getLogger(__name__)

def search_gpu(X, directions, dx, reference_time, num_results=10):
    from .gpu_impl import projected_bounds, _vote_points, _vote_points_mask, _find_voters_points, _hough_max
    n = X.shape[0]
    
    x_min, x_max, y_min, y_max = projected_bounds(X, directions.b, reference_time)
    
    num_dir = directions.b.shape[0]
    _dx = dx.to(u.deg).value
    _dy = _dx
    num_x = int((x_max - x_min) / _dx  + 1)
    num_y = int((y_max - y_min) / _dy  + 1)

    log.info("creating hough space with shape (%d, %d, %d)", num_dir, num_x, num_y)
    hough = np.zeros((num_dir, num_x, num_y), dtype=np.int32)

    num_dir, num_x, num_y = hough.shape
    
    mask = np.full((num_dir, n), False)
    
    d_hough = cuda.to_device(hough)
    d_mask = cuda.to_device(mask)
    d_X = cuda.to_device(X)
    d_directions = cuda.to_device(directions.b)
    d_max = cuda.to_device(np.zeros((hough.shape[0], 3), dtype=np.int32))
    d_results = cuda.to_device(np.zeros((num_results, 4), dtype=np.int32))

    def _vote(coef=1):
        # Configure GPU threads and blocks
        threads_per_block = (16, 16)  # Tunable parameters
        blocks_per_grid = ((num_dir + threads_per_block[0] - 1) // threads_per_block[0], 
                           (n + threads_per_block[1] - 1) // threads_per_block[1])

        # Launch the CUDA kernel
        _vote_points[blocks_per_grid, threads_per_block](
            d_hough, d_X, d_directions, x_min, y_min, _dx, _dy, reference_time, coef
        )

    def _vote_mask(coef=1):
        # Configure GPU threads and blocks
        threads_per_block = (16, 16)  # Tunable parameters
        blocks_per_grid = ((num_dir + threads_per_block[0] - 1) // threads_per_block[0], 
                           (n + threads_per_block[1] - 1) // threads_per_block[1])

        # Launch the CUDA kernel
        _vote_points_mask[blocks_per_grid, threads_per_block](
            d_hough, d_mask, d_X, d_directions, x_min, y_min, _dx, _dy, reference_time, coef
        )
    
    def _max():
        _hough_max[256, num_dir // 256 + 1](d_max, d_hough)
        
    def _find(mask_dir, mask_x, mask_y):
        threads_per_block = (16, 16)  # Tunable parameters
        blocks_per_grid = ((num_dir + threads_per_block[0] - 1) // threads_per_block[0], 
                           (n + threads_per_block[1] - 1) // threads_per_block[1])

        _find_voters_points[blocks_per_grid, threads_per_block](
            d_X, d_directions, d_mask, x_min, y_min, _dx, _dy, reference_time, mask_dir, mask_x, mask_y
        )

    _vote(coef=1)
    # results = [] # cuda device array
    for n_i in range(num_results):
        _max()
        i = -1
        v = -np.inf
        for _ in range(len(d_max)):
            if d_max[_, 2] > v:
                v = d_max[_, 2]
                i = _
        j = d_max[i, 0]
        k = d_max[i, 1]
        d_results[n_i, 0] = i
        d_results[n_i, 1] = j
        d_results[n_i, 2] = k
        d_results[n_i, 3] = v
        _find(i, j, k)
        print("cluster has value", v, "at", (i, j, k))
        _vote_mask(coef=-1)
    
    return d_results.copy_to_host()

def search(X, directions, dx, reference_time, num_results=10, precompute=False, gpu=False):
    if gpu:
        from .gpu_impl import projected_bounds, hough_max, make_bins, vote_points, vote_bins, find_voters_points, find_voters_bins
    else:
        from .cpu_impl import projected_bounds, hough_max, make_bins, vote_points, vote_bins, find_voters_points, find_voters_bins

    def find_clusters_points(X, hough, directions, x_min, y_min, dx, dy, reference_time, n=10):
        results = np.full((n, 4), -1)
        results_points = []
        include = np.full(X.shape[0], True)
        for i in range(n):
            idx, val = hough_max(hough)
            print("cluster has value", val, "at", idx)
            voters = find_voters_points(
                hough, X, directions.b, x_min, y_min, dx, dy, reference_time, *idx
            )
            mask = include & voters
            hough = vote_points(
                hough, X[mask], directions.b, x_min, y_min, dx, dy, reference_time, -1
            )
            include &= ~voters # exclude voters
            print(include.sum(), "/", X.shape[0], "points remain")
            results[i, 0] = idx[0]
            results[i, 1] = idx[1]
            results[i, 2] = idx[2]
            results[i, 3] = val
            results_points.append(X[mask])
            
        return results, results_points

    def find_clusters_bins(X, bins, hough, n=10):
        results = np.full((n, 4), -1)
        results_points = []
        include = np.full(bins.shape[0], True)
        for i in range(n):
            idx, val = hough_max(hough)
            print("cluster has value", val, "at", idx)
            voters = find_voters_bins(
                hough, bins, *idx
            )
            mask = include & voters
            hough = vote_bins(
                hough, bins[mask], -1
            )
            include &= ~voters # exclude voters
            print(include.sum(), "/", X.shape[0], "points remain")
            results[i, 0] = idx[0]
            results[i, 1] = idx[1]
            results[i, 2] = idx[2]
            results[i, 3] = val
            results_points.append(X[mask])
            
        return results, results_points

    n = X.shape[0]
    
    x_min, x_max, y_min, y_max = projected_bounds(X, directions.b, reference_time)
    
    num_dir = directions.b.shape[0]
    _dx = dx.to(u.deg).value
    _dy = _dx
    num_x = int((x_max - x_min) / _dx  + 1)
    num_y = int((y_max - y_min) / _dy  + 1)

    log.info("creating hough space with shape (%d, %d, %d)", num_dir, num_x, num_y)
    hough = np.zeros((num_dir, num_x, num_y), dtype=np.uint32)    
    
    if precompute:
        bins = make_bins(X, directions.b, x_min, y_min, _dx, _dy, reference_time)
        hough = vote_bins(hough, bins, 1)
        results, results_points = find_clusters_bins(X, bins, hough, n=num_results)
    else:
        hough = vote_points(hough, X, directions.b, x_min, y_min, _dx, _dy, reference_time, 1)
        results, results_points = find_clusters_points(X, hough, directions, x_min, y_min, _dx, _dy, reference_time, n=num_results)
    
    return results, results_points

def main():
    import argparse
    parser = argparse.ArgumentParser(prog=__name__)
    parser.add_argument("--psfs", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--dx", required=True, type=float)
    parser.add_argument("--velocity", required=True, nargs=2, type=float)
    parser.add_argument("--angle", required=True, nargs=2, type=float)
    parser.add_argument("--threshold", required=True, type=int)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--precompute", action='store_true')
    parser.add_argument("--gpu", action='store_true')
    parser.add_argument("--gpu-kernels", action='store_true')
    parser.add_argument("--device", type=int, required=False, default=-1)

    args = parser.parse_args()

    if args.gpu and args.device > -1:
        cuda.select_device(args.device)
    
    psfs = astropy.table.Table.read(args.psfs)['psf']
    log.info("seeing [min, median, max]: [%f, %f, %f]", np.min(psfs), np.median(psfs), np.max(psfs))

    dx = args.dx * np.median(psfs) * psfs.unit
    log.info(f"using dx = {dx}")
    catalog = astropy.table.Table.read(args.catalog)

    X = np.array([catalog['ra'], catalog['dec'], catalog['time']]).T
    reference_epoch = X[:, 2].min() * u.day
    dt = (X[:, 2].max() - X[:, 2].min()) * u.day
    vmin = args.velocity[0] * u.deg/u.day
    vmax = args.velocity[1] * u.deg/u.day
    phimin = args.angle[0] * u.deg
    phimax = args.angle[1] * u.deg

    directions = SearchDirections([vmin, vmax], [phimin, phimax], dx, dt)
    log.info("searching %d directions", len(directions.b))
    if args.gpu_kernels:
        results, results_points = search_gpu(X, directions, dx, reference_epoch.value, num_results=args.threshold)
    else:
        results, results_points = search(X, directions, dx, reference_epoch.value, num_results=args.threshold, precompute=args.precompute, gpu=args.gpu)

    args.results_dir.mkdir(parents=True, exist_ok=False)
    for i, (result, points) in enumerate(zip(results, results_points)):
        # refine
        try:
            refined = refine(points)
            gathered = gather(refined['result'], X[:, 0], X[:, 1], X[:, 2], 1/3600)
        except Exception as e:
            log.error(str(e))
            continue
        d = args.results_dir / str(i)
        d.mkdir(parents=True, exist_ok=True)
        r = refined['result']
        l = refined['line']
        astropy.table.Table(
            [
                {
                    "x": result[0],
                    "y": result[1],
                    "direction": result[2],
                    "n": result[3],
                }
            ]
        ).write(d / "result.ecsv", format="ascii.ecsv")

        reference_sky_pos = l.predict(reference_epoch)
        astropy.table.Table(
            [
                {
                    "vra": r.beta[0, 0] * u.deg/u.day,
                    "vdec": r.beta[0, 1] * u.deg/u.day,
                    "ra_0": r.alpha[0] * u.deg,
                    "dec_0": r.alpha[1] * u.deg,
                    "ra_ref": reference_sky_pos[0][0],
                    "dec_ref": reference_sky_pos[0][1],
                    "tref": reference_epoch,
                    "tmin": points[:, 2].min() * u.day,
                    "tmax": points[:, 2].max() * u.day,
                    "sigma_vra": r.sigma_e[0, 0]**0.5 * u.deg/u.day,
                    "sigma_vdec": r.sigma_e[1, 1]**0.5 * u.deg/u.day,
                    "sigma_vravdec": r.sigma_e[0, 1] * (u.deg/u.day)**2,
                    "sigma_vdecvra": r.sigma_e[1, 0] * (u.deg/u.day)**2,
                    "sigma_t": r.sigma_xx[0, 0] * u.day,
                }
            ]
        ).write(d / "tracklet.ecsv", format="ascii.ecsv")

        t = astropy.table.Table(
            [
                {
                    "ra": p[0] * u.deg,
                    "dec": p[1] * u.deg,
                    "time": p[2] * u.day,
                }
                for p in points
            ]
        )
        t.sort("time")
        t.write(d / "points.ecsv", format="ascii.ecsv")

        t = catalog[gathered]
        t.sort("time")
        t.write(d / "gathered.ecsv", format="ascii.ecsv")


if __name__ == "__main__":
    main()
