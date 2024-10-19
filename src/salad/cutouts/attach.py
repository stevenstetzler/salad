from salad.serialize import read, write
import logging
from .get import times_for

logging.basicConfig()
log = logging.getLogger(__name__)

def attach_cutouts_from_exposures(clusters, exposures, width, height):
    import lsst.geom
    exp_times = times_for(exposures)
    for i, k in enumerate(clusters):
        cluster = clusters[k]
        log.info("getting cutouts for cluster %d %d/%d", k, i, len(clusters))
        locations = cluster.line.predict(exp_times)
        cutouts = {}
        centers = {}
        for (ra, dec), expnum in zip(locations, exposures):
            exposure = exposures[expnum]
            sp = lsst.geom.SpherePoint(ra.value, dec.value, lsst.geom.degrees)
            p = exposure.wcs.skyToPixel(sp)
            centers[expnum] = {
                "pix": p,
                "wcs": sp,
            }
            bbox = lsst.geom.Box2I(lsst.geom.Point2I(p.getX() - width/2, p.getY() - height/2), lsst.geom.Extent2I(width, height))
            bbox.clip(exposure.getBBox())
            if bbox.getWidth() == width and bbox.getHeight() == height:
                cutout = exposure.getCutout(bbox)
                cutouts[expnum] = cutout

        log.info("cluster %d has %d/%d cutouts", i, len(cutouts), len(exposures))
        cluster.cutouts = cutouts
        cluster.centers = centers

def main():
    import argparse
    import sys
    import os

    parser = argparse.ArgumentParser(prog=__name__)

    parser.add_argument("clusters", nargs='?', type=str)
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--width", type=int, default=50)
    parser.add_argument("--height", type=int, default=50)
    args = parser.parse_args()

    images = read(args.images)
    exposures = {}
    for i, image in enumerate(images):
        exposures[image.expnum] = image.exposure

    for path in args.clusters:
        clusters = read(path)
        attach_cutouts_from_exposures(clusters, exposures, args.width, args.height)
        write(
            clusters, 
            os.path.join(
                os.path.dirname(path), 
                os.path.basename(path).replace(".pkl", "_cutouts.pkl")
            )
        )
        del clusters

if __name__ == "__main__":
    main()