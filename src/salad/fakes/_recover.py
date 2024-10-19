import numpy as np
import joblib
import os
from astropy.coordinates import SkyCoord
import astropy.units as u

def recover_fake(orbit_id, pg_fakes, search_exposures, points, results, match_threshold):
    search_separations = dict()
    search_detect = dict()
    result_separations = dict()
    result_detect = dict()
    
    fake = pg_fakes[pg_fakes['ORBITID'] == orbit_id]
    fake_coords = SkyCoord(fake['RA'] * u.deg, fake['DEC'] * u.deg)

    search_separations[orbit_id] = []
    search_detect[orbit_id] = 0

    result_separations[orbit_id] = [[] for result in results]
    result_detect[orbit_id] = [0 for result in results]
    
    for i, expnum in enumerate(fake['EXPNUM']):
        fake_coord = fake_coords[i]        
        search_mask = search_exposures == expnum
        search_points = points[search_mask]
        search_coords = SkyCoord(search_points[:, 0] * u.deg, search_points[:, 1] * u.deg)
        
        search_separation = fake_coord.separation(search_coords)
        search_match = (search_separation < match_threshold).sum()
        
        search_separations[orbit_id].append(search_separation)
        search_detect[orbit_id] += search_match
        for j, result in enumerate(results):
            
            result_mask = search_mask & result.close
            result_points = points[result_mask]
            result_coords = SkyCoord(result_points[:, 0] * u.deg, result_points[:, 1] * u.deg)
            result_separation = fake_coord.separation(result_coords)
            result_match = (result_separation < match_threshold).sum()

            result_separations[orbit_id][j].append(result_separation)
            result_detect[orbit_id][j] += result_match
            
    return search_separations, search_detect, result_separations, result_detect

def did_find(catalog, search, recover_results):
    import numpy as np
    
    orbits = recover_results['search']['detections'].keys()

    result_metrics = {}
    possible_to_find = {}
    did_find = {}
    for orbit in orbits:
        possible_orbit_detections = recover_results['search']['detections'][orbit]

        if possible_orbit_detections < (catalog.num_times / 2):
            possible_to_find[orbit] = (False, possible_orbit_detections)
            print(orbit, "cannot be found", possible_orbit_detections, (catalog.num_times / 2))
            continue

        possible_to_find[orbit] = (True, possible_orbit_detections)

        orbit_detections = recover_results['result']['detections'][orbit]
        result_metrics[orbit] = []
        purities = []
        for i, result in enumerate(search.results):
            num_fake_detections = orbit_detections[i]
            num_result_detections = result.close.sum()
            purity = num_fake_detections / num_result_detections
            accuracy = num_fake_detections / possible_orbit_detections
            result_metrics[orbit].append(
                dict(
                    purity=purity,
                    accuracy=accuracy,
                    possible_orbit_detections=possible_orbit_detections,
                    num_fake_detections=num_fake_detections,
                    num_result_detections=num_result_detections, 
                )
            )
            purities.append(purity)

        did_find[orbit] = np.where(np.array(purities) > 0.5)[0]

    return did_find, possible_to_find

def merge_dicts(dicts):
    data = {}
    for d in dicts:
        data.update(d)
    return data

def main():
    import argparse
    import pickle
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=argparse.FileType('rb'), default=sys.stdin)
    parser.add_argument('output', nargs='?', type=argparse.FileType('wb'), default=sys.stdout)
    parser.add_argument("search", type=str)
    parser.add_argument("catalog", type=str)
    parser.add_argument("--match-threshold", type=int, default=1)
    parser.add_argument("--match-threshold-units", default="arcsec")
    parser.add_argument("--where", type=str, default="instrument='DECam'")
    parser.add_argument("--processes", "-J", type=int, default=1)


    args = parser.parse_args()

    if args.input == sys.stdin:
        args.input = sys.stdin.buffer
    
    fakes = pickle.load(args.input)

    match_threshold = args.match_threshold * getattr(u, args.match_threshold_units)

    with open(os.path.join(args.search), "rb") as fd:
        search = pickle.load(fd)
        
    with open(os.path.join(args.catalog), "rb") as fd:
        catalog = pickle.load(fd)

    search_exposures = catalog.exposure
    orbit_ids = set(fakes['ORBITID'])

    print("recovering", len(orbit_ids), "fakes in", len(search.results), "results and", len(search_exposures), "points", file=sys.stderr)
    recover_results = joblib.Parallel(
        n_jobs=args.processes
    )(
        joblib.delayed(recover_fake)(orbit_id, fakes, search_exposures, search.hough.X, search.results, match_threshold)
        for orbit_id in orbit_ids
    )

    search_separations = merge_dicts([x[0] for x in recover_results])
    search_detect = merge_dicts([x[1] for x in recover_results])
    result_separations = merge_dicts([x[2] for x in recover_results])
    result_detect = merge_dicts([x[3] for x in recover_results])
        
    recover_results = {
        "search": {
            "separations": search_separations,
            "detections": search_detect,
        },
        "result": {
            "separations": result_separations,
            "detections": result_detect
        }
    }

    recover_results['did_find'], recover_results['possible_to_find'] = did_find(catalog, search, recover_results)

    if args.output is sys.stdout:
        args.output = sys.stdout.buffer
    pickle.dump(recover_results, args.output)

if __name__ == "__main__":
    main()
