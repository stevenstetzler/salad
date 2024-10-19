from ..serialize import read, write
import logging
from .fitting import fit_trajectory

logging.basicConfig()
log = logging.getLogger(__name__)

def get_positions(cluster, times):
    line = cluster.extra['line']
    positions = (times[:, None] @ line.beta) + line.alpha
    return positions

def main():
    import argparse
    import sys
    import astropy.time
    import numpy as np
    import astropy.units as u
    import lsst.geom

    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs='?', type=argparse.FileType('rb'), default=sys.stdin)
    parser.add_argument('output', nargs='?', type=argparse.FileType('wb'), default=sys.stdout)
    parser.add_argument("--images", type=str, required=True)

    args = parser.parse_args()

    cluster = read(args.input)
    images = read(args.images)
    images = [image for image in images]
    times = []
    cutouts = []
    for image in images:
        visit_info = image.reader.readExposureInfo().getVisitInfo()
        time = visit_info.date.toAstropy() + astropy.time.TimeDelta(visit_info.exposureTime / 2 + 0.5, format='sec')
        times.append(time.value)
    times = np.array(times) * u.day

    positions = get_positions(cluster, times)
    cutouts = []
    clipped = []
    points = []
    cutout_width = 50
    cutout_height = 50
    for position, image in zip(positions, images):
        wcs = image.reader.readWcs()
        exposure_bbox = image.reader.readBBox()
        point = wcs.skyToPixel(lsst.geom.SpherePoint(position[0].to(u.deg).value, position[1].to(u.deg).value, units=lsst.geom.degrees))
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(point.getX() - cutout_width/2, point.getY() - cutout_height/2), lsst.geom.Extent2I(cutout_width, cutout_height))
        bbox.clip(exposure_bbox) # clip to fit the cutout in the exposure
        clipped.append((bbox.width != cutout_width) or (bbox.height != cutout_height)) # indicate if the box was clipped off; this might be picked up as a EDGE mask too...
        cutouts.append(image.reader.read(bbox))
        points.append(point)

    line = cluster.extra['line']
    trajectory_fit = fit_trajectory(cutouts, points, times.value, lambda p, t : (p[0] * t + p[2], p[1] * t + p[3]), start=np.hstack([line.beta.value.flatten(), line.alpha.value.flatten()]))
    cluster.extra['fit'] = trajectory_fit
    cluster.extra.pop("cutouts", None)
    write(cluster, args.output)

if __name__ == "__main__":
    main()
