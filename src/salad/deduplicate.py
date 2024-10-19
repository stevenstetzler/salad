import numpy as np
import logging
from .serialize import read, write

logging.basicConfig()
log = logging.getLogger(__name__)

def match_distance(a):
    d = np.sum(
        np.vstack(
            [
                [np.subtract.outer(a[:, 0], a[:, 0])**2],
                [np.subtract.outer(a[:, 1], a[:, 1])**2]
            ]
        ), 
        axis=0
    )**0.5
    return d
    
    
def quantities(clusters, time_zero):
    origins = []
    alphas = []
    betas = []
    idx_to_keys = {}
    keys_to_idx = {}
    for i, k in enumerate(clusters):
        origins.append(np.atleast_2d(clusters[k].line.predict(time_zero).value)[0])
        alphas.append(clusters[k].line.alpha.value)
        betas.append(np.atleast_2d(clusters[k].line.beta.value)[0])
        idx_to_keys[i] = k
        keys_to_idx[k] = i
    
    origins = np.array(origins)
    alphas = np.array(alphas)
    betas = np.array(betas)

    return origins, alphas, betas, idx_to_keys, keys_to_idx
    
def find_matches(distances, threshes):
    distances = np.atleast_3d(distances)
    threshes = np.atleast_1d(threshes)
    matches = {}
    matched = []

    for i in range(distances[-1].shape[0]):
        masks = []
        for j, t in enumerate(threshes):
            masks.append(distances[i, :, j] < t)
            
        mask = np.logical_and.reduce(masks)
        idx = np.where(mask)[0]
        idx = idx[idx != i]
        if len(idx) > 0:
            m = []
            for j in idx:
                if j not in matches and j not in matched:
                    m.append(j)
                    matched.append(j)
            if len(m) > 0:
                matches[i] = m
    
    return matches    

def deduplicate(clusters, time_zero, origin_thresh, beta_thresh):
    origins, alphas, betas, idx_to_keys, keys_to_idx = quantities(clusters, time_zero)
    
    sky_units = clusters[next(iter(clusters.keys()))].line.alpha.unit
    velocity_units = clusters[next(iter(clusters.keys()))].line.beta.unit

    matches = find_matches(
        np.array([match_distance(origins), match_distance(betas)]).T, 
        [origin_thresh.to(sky_units).value, beta_thresh.to(velocity_units).value]
    )
    dedup = {k: clusters[k] for k in clusters}

    duplicates = {}
    for k in clusters:
        if keys_to_idx[k] in matches:
            for j in matches[keys_to_idx[k]]:
                if idx_to_keys[j] in dedup:
                    duplicates[k] = idx_to_keys[j]
                    dedup.pop(idx_to_keys[j])
    return dedup, duplicates

def main():
    import argparse
    import sys
    import astropy.units as u

    parser = argparse.ArgumentParser(prog=__name__)

    parser.add_argument("input", nargs="?", type=argparse.FileType('rb'), default=sys.stdin)
    parser.add_argument('output', nargs='?', type=argparse.FileType('wb'), default=sys.stdout)
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--origin-threshold", type=float, default=5)
    parser.add_argument("--origin-threshold-units", default="arcsec", type=str)
    parser.add_argument("--beta-threshold", type=int, default=1)
    parser.add_argument("--beta-threshold-units", default="arcsec", type=str)
    
    args = parser.parse_args()
    clusters = read(args.input)
    images = read(args.images)

    times = [image.mjd_mid for image in images] * u.day

    time_zero = min(times)
    time_max = max(times)
    log.info("deduplicating %d clusters", len(clusters))
    dedup, duplicates = deduplicate(
        clusters, 
        time_zero, 
        (args.origin_threshold * getattr(u, args.origin_threshold_units)), 
        ((args.beta_threshold * getattr(u, args.beta_threshold_units))/(time_max - time_zero))
    )
    log.info("there are %d deduplicated clusters", len(dedup))
    write(dedup, args.output)

if __name__ == "__main__":
    main()
