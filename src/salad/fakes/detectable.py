import logging
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from ..serialize import read, write

logging.basicConfig()
log = logging.getLogger(__name__)

def detections_in_catalog(fakes, catalog, match_threshold=1 * u.arcsec):
    exposures = set(fakes['EXPNUM'])
    orbits = set(fakes['ORBITID'])
    catalog_coords = SkyCoord(catalog.ra, catalog.dec)
    catalog_exposures = catalog.exposure
    recoveries = {orbit: 0 for orbit in orbits}
    for exposure in exposures:
        exposure_catalog_coords = catalog_coords[catalog_exposures == exposure]
        exposure_fakes = fakes[fakes['EXPNUM'] == exposure]
        exposure_fakes_coords = SkyCoord(exposure_fakes['RA'] * u.deg, exposure_fakes['DEC'] * u.deg)
        for fake, fake_coord in zip(exposure_fakes, exposure_fakes_coords):
            sep = fake_coord.separation(exposure_catalog_coords)
            matches = sep < match_threshold
            num_matches = matches.sum()
            if num_matches > 0:
                recoveries[fake['ORBITID']] += num_matches
                
    return recoveries
    
def velocity_in_hough(fakes, hough):
    b = hough.projection.directions.b
    recoveries = {}
    for orbit in fakes.group_by("ORBITID").groups:
        dt = (orbit['mjd_mid'][-1] - orbit['mjd_mid'][0]) * u.day

        if dt == 0:
            vra = np.nan
            vdec = np.nan
        else:
            dra = (orbit['RA'][-1] - orbit['RA'][0]) * u.deg
            ddec = (orbit['DEC'][-1] - orbit['DEC'][0]) * u.deg
            vra = dra / dt
            vdec = ddec / dt
        
        dv = b - np.array([vra.value, vdec.value]) * u.deg/u.day
        min_dv = ((dv**2).sum(axis=1)**0.5).min()
        distance = min_dv * dt        
        recoveries[orbit[0]['ORBITID']] = (distance < hough.dx * u.deg)
    
    return recoveries

def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser(prog=__name__)
    parser.add_argument("fakes", type=str)
    parser.add_argument("catalog", type=str)
    parser.add_argument("hough", type=str)
    parser.add_argument('output', nargs='?', type=argparse.FileType('wb'), default=sys.stdout)
    parser.add_argument("--match-threshold", type=float, default=1)
    parser.add_argument("--match-threshold-unit", type=str, default="arcsec")
    
    args = parser.parse_args()

    fakes = read(args.fakes)
    catalog = read(args.catalog)
    hough = read(args.hough)
    threshold = args.match_threshold * getattr(u, args.match_threshold_unit)

    findable_detections = detections_in_catalog(fakes, catalog, )
    findable_velocity = velocity_in_hough(fakes, hough)

    write(recoveries, args.output)
