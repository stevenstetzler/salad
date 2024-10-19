import logging
import sys
from .serialize import Serializable, read, write

logging.basicConfig()
log = logging.getLogger(__name__)

class Image(Serializable):
    def __init__(self, dataset, path):
        self.dataset = dataset
        self.path = path

    @property
    def exposure(self):
        if not hasattr(self, "_exposure"):
            self._exposure = self.read()

        return self._exposure

    @property
    def exposureInfo(self):
        if not hasattr(self, "_exposureInfo"):
            self._exposureInfo = self.reader.readExposureInfo()
        return self._exposureInfo
    
    @property
    def visitInfo(self):
        if not hasattr(self, "_visitInfo"):
            self._visitInfo = self.exposureInfo.getVisitInfo()
        return self._visitInfo

    @property
    def expnum(self):
        if not hasattr(self, "_expnum"):
            self._expnum = self.visitInfo.id
        return self._expnum

    @property
    def exposureTime(self):
        if not hasattr(self, "_exposureTime"):
            self._exposureTime = self.visitInfo.getExposureTime()
        return self._exposureTime

    # @property
    # def exposureDate(self):
    #     import astropy.time
    #     if not hasattr(self, "_exposureDate"):
    #         self._exposureDate = (self.visitInfo.date.toAstropy() + astropy.time.TimeDelta(self.visitInfo.exposureTime / 2 + 0.5, format='sec')).value
    #     return self._exposureTime

    @property
    def mjd_mid(self):
        import astropy.time
        if not hasattr(self, "_mjd_mid"):
            self._mjd_mid = (self.visitInfo.date.toAstropy() + astropy.time.TimeDelta(self.visitInfo.exposureTime / 2 + 0.5, format='sec')).value
        return self._mjd_mid

    @property
    def reader(self):
        import lsst.afw.image
        if not hasattr(self, "_reader"):
            self._reader = lsst.afw.image.ExposureFitsReader(self.path)
        return self._reader

    def read(self, *args, **kwargs):
        import lsst.afw.image
        log.info("reading %s", self.path)
        return lsst.afw.image.ExposureFitsReader(self.path).read(*args, **kwargs)

    def __reduce__(self):
        return type(self), (self.dataset, self.path, )
    
def main():
    import argparse
    import lsst.daf.butler as dafButler
    parser = argparse.ArgumentParser(prog=__name__)
    parser.add_argument("repo", type=str)
    parser.add_argument("datasetType", type=str)
    parser.add_argument('output', nargs='?', type=argparse.FileType('wb'), default=sys.stdout)
    parser.add_argument("--collections", type=str, default="*")
    parser.add_argument("--where", type=str, default="instrument='DECam'")

    args = parser.parse_args()

    butler = dafButler.Butler(args.repo)
    collections = butler.registry.queryCollections(args.collections)
    collections = list(set(collections))
    log.info(f"searching for {args.datasetType} in {collections} where {args.where}")
    refs = list(set(list(butler.registry.queryDatasets(args.datasetType, where=args.where, collections=collections))))
    log.info(f"found {len(refs)} of type {args.datasetType} in {collections} where {args.where}")
    images = [Image(args.datasetType, butler.getURI(ref).ospath) for ref in refs]
    write(images, args.output)

if __name__ == "__main__":
    main()
