import logging
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from ..serialize import read, write

logging.basicConfig()
log = logging.getLogger(__name__)

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
        recoveries[orbit[0]['ORBITID']] = dict(
            vra=vra,
            vdec=vdec,
            min_dv=min_dv,
            distance=distance,
            findable=(distance < hough.dx * u.deg),
        )
    
    return recoveries

def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser(prog=__name__)
    parser.add_argument("fakes", type=str)
    parser.add_argument("hough", type=str)
    parser.add_argument('output', nargs='?', type=argparse.FileType('wb'), default=sys.stdout)
    
    args = parser.parse_args()

    fakes = read(args.fakes)
    hough = read(args.hough)

    findable = velocity_in_hough(fakes, hough)
    write(findable, args.output)

if __name__ == "__main__":
    main()

