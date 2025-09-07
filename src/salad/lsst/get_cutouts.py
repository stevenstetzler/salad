from ..io.serialize import read, write
from functools import lru_cache
import logging
import numpy as np
import lsst.geom
import astropy.time
import astropy.units as u

logging.basicConfig()
log = logging.getLogger(__name__)

def attach_cutouts_from_images(clusters, images):
    exp_times = np.array(list(map(lambda x : x.mjd_mid, images))) * u.day
    wcss = list(map(lambda x : x.reader.readWcs(), images))
    bboxes = list(map(lambda x : x.reader.readBBox(), images))
    for i, cluster in enumerate(clusters):
        log.info("getting cutouts for cluster %d/%d", i, len(clusters))
        locations = cluster.line.predict(exp_times)
        cutouts = {}
        centers = {}
        for (ra, dec), image, wcs, image_bbox in zip(locations, images, wcss, bboxes):
            sp = lsst.geom.SpherePoint(ra.value, dec.value, lsst.geom.degrees)
            p = wcs.skyToPixel(sp)
            expnum = image.visitInfo.id
            centers[expnum] = {
                "pix": p,
                "wcs": sp,
            }
            bbox = lsst.geom.Box2I(lsst.geom.Point2I(p.getX() - 25, p.getY() - 25), lsst.geom.Extent2I(50, 50))
            bbox.clip(image_bbox)
            if bbox.getWidth() == 50 and bbox.getHeight() == 50:
                cutout = image.reader.read(bbox)
                cutouts[expnum] = cutout
        log.info("cluster %d has %d/%d cutouts", i, len(cutouts), len(images))
        cluster.cutouts = cutouts
        cluster.centers = centers
        
def times_for(exposures):
    times = []
    for e in exposures:
        times.append((exposures[e].visitInfo.date.toAstropy() + astropy.time.TimeDelta(exposures[e].visitInfo.exposureTime / 2 + 0.5, format='sec')).value)
    return np.array(times) * u.day
        
def attach_cutouts_from_exposures(clusters, exposures):
    exp_times = times_for(exposures)
    for i in clusters:
        cluster = clusters[i]
        log.info("getting cutouts for cluster %d/%d", i, len(clusters))
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
            bbox = lsst.geom.Box2I(lsst.geom.Point2I(p.getX() - 25, p.getY() - 25), lsst.geom.Extent2I(50, 50))
            bbox.clip(exposure.getBBox())
            if bbox.getWidth() == 50 and bbox.getHeight() == 50:
                cutout = exposure.getCutout(bbox)
                cutouts[expnum] = cutout

        log.info("cluster %d has %d/%d cutouts", i, len(cutouts), len(exposures))
        cluster.cutouts = cutouts
        cluster.centers = centers
        

@lru_cache(maxsize=512)
def _cutout(butler, ref, x, y, w, h):
    import lsst.geom
    corner = lsst.geom.Point2I(x - w/2, y - h/2)
    bbox = lsst.geom.Box2I(corner, lsst.geom.Extent2I(w, h))
    log.info("getting ref %s with bbox %s", ref, bbox)
    return butler.get(ref, parameters=dict(bbox=bbox))

def get_cutouts(butler, refs, points, w=50, h=50):
    cutouts = []
    for row in points:
        visit = row['expnum']
        i_x, i_y = row['i_x'], row['i_y']
        ref = refs[visit]
        cutout = _cutout(butler, ref, i_x, i_y, w, h)
        cutouts.append(cutout)
    return cutouts

def main():
    import argparse
    import lsst.daf.butler as dafButler
    import sys

    parser = argparse.ArgumentParser()
    # inputs: catalog clusters
    parser.add_argument("input", nargs='?', type=argparse.FileType('rb'), default=sys.stdin)
    parser.add_argument('output', nargs='?', type=argparse.FileType('wb'), default=sys.stdout)
    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument("--datasetType", type=str, required=True)
    parser.add_argument("--collections", type=str, default="*")
    parser.add_argument("--where", type=str, default="instrument='DECam'")
    parser.add_argument("--width", type=int, default=50)
    parser.add_argument("--height", type=int, default=50)
    args = parser.parse_args()

    cluster = read(args.input)
    p = cluster.extra['join']

    butler = dafButler.Butler(args.repo)

    datasetType = args.datasetType
    collections = butler.registry.queryCollections(args.collections)
    where = args.where + " and visit in (" + ",".join(map(str, map(int, set(p['expnum'])))) + ")"
    log.info("getting cutouts from %s in %s where %s", datasetType, collections, where)
    refs = butler.registry.queryDatasets(
        datasetType,
        collections=collections,
        where=where,
    )
    refs = {ref.dataId['visit'] : ref for ref in refs}
    log.info("getting %s cutouts", len(p))
    cutouts = get_cutouts(butler, refs, p, args.width, args.height)
    cluster.extra['cutouts'] = cutouts
    write(cluster, args.output)

if __name__ == "__main__":
    main()
