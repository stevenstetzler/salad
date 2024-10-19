import astropy.table
import astropy.time
import numpy as np
import pickle
import astropy.units as u
from .serialize import Serializable


class DetectionCatalog(Serializable):
    def __init__(self, catalog : astropy.table.Table, time: astropy.time.Time, exposure: int, detector: int = None, masked_pixel_summary: dict = None):
        self.catalog = catalog
        self._time = time
        self.exposure = exposure
        self.detector = detector
        self.time = time.mjd * u.day
        self.masked_pixel_summary = masked_pixel_summary

    def _column_factory(column):
        return lambda self: self.catalog[column]
    
    def _column_factory(column):
        def _get(self):
            if column == "times":
                return astropy.table.Column([self.time.value] * self.size, unit=u.day, name="times")
            elif column in "exposures":
                return astropy.table.Column([self.exposure] * self.size, name="exposures")
            elif column in "detectors":
                return astropy.table.Column([self.detector] * self.size, name="detectors")
            elif column == "mask_summary":
                return astropy.table.Table([{"exposure": self.exposure, "detector": self.detector, **self.masked_pixel_summary}])
            else:
                return self.catalog[column]
                
        return _get


    ra = property(_column_factory("ra"))
    dec = property(_column_factory("dec"))
    peakValue = property(_column_factory("peakValue"))
    significance = property(_column_factory("significance"))
    i_x = property(_column_factory("i_x"))
    i_y = property(_column_factory("i_y"))
    times = property(_column_factory("times"))
    exposures = property(_column_factory("exposures"))
    detectors = property(_column_factory("detectors"))
    mask_summary = property(_column_factory("mask_summary"))

    @property
    def size(self):
        return len(self.catalog)

    def __reduce__(self):
        return type(self), (
            self.catalog,
            self._time,
            self.exposure,
            self.detector,
            self.masked_pixel_summary,
        )

class MultiEpochDetectionCatalog(Serializable):
    def __init__(self, single_epoch_catalogs):
        self.single_epoch_catalogs = single_epoch_catalogs

    def _column_factory(column):
        def _get(self):
            t = astropy.table.vstack([getattr(c, column) for c in self.single_epoch_catalogs])
            if column == "mask_summary":
                return t
            return t[column]
                
        return _get

    @property
    def num_times(self):
        return len(self.single_epoch_catalogs)

    time = property(_column_factory("times"))
    exposure = property(_column_factory("exposures"))
    detector = property(_column_factory("detectors"))
    ra = property(_column_factory("ra"))
    dec = property(_column_factory("dec"))
    peakValue = property(_column_factory("peakValue"))
    significance = property(_column_factory("significance"))
    i_x = property(_column_factory("i_x"))
    i_y = property(_column_factory("i_y"))
    mask_summary = property(_column_factory("mask_summary"))

    # @property
    # def time(self):
    #     return astropy.table.Column([c.time.value for c in self.single_epoch_catalogs], unit=u.day)

    # @property
    # def ra(self):
    #     return astropy.table.vstack([c.catalog["ra"] for c in self.single_epoch_catalogs])['ra']
    
    # @property
    # def dec(self):
    #     return astropy.table.vstack([c.catalog["dec"] for c in self.single_epoch_catalogs])
    
    def X(self, columns=None, sky_units=u.deg, time_units=u.day):
        if columns is None:
            columns = ['ra', 'dec', 'times', 'exposures']

        xs = [[] for c in columns]
        for c in self.single_epoch_catalogs:
            for i, col in enumerate(columns):
                d = getattr(c, col)
                if col in ["ra", "dec"]:
                    d = d.to(sky_units)
                elif col == "time":
                    d = astropy.table.Column((([c.time.value] * c.size)*u.day).to(time_units))
                xs[i].append(d.data)
        
        xs = [np.hstack(x) for x in xs]
        return np.array(xs).T

    def __reduce__(self):
        return type(self), (
            self.single_epoch_catalogs,
        )
    

def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=argparse.FileType('rb'), default=sys.stdin)
    parser.add_argument('output', nargs='?', type=argparse.FileType('wb'), default=sys.stdout)
    parser.add_argument("--columns", nargs="+", default=None)
    parser.add_argument("--sky-units", default="deg")
    parser.add_argument("--time-units", default="day")
    args = parser.parse_args()

    try:
        catalog = MultiEpochDetectionCatalog.read(args.input)
    except:
        catalog = DetectionCatalog.read(args.input)

    try:
        np.savetxt(args.output, catalog.X(columns=args.columns, sky_units=getattr(u, args.sky_units), time_units=getattr(u, args.time_units)))
    except BrokenPipeError:
        pass


if __name__ == "__main__":
    main()
