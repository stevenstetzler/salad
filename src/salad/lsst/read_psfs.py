

def main():
    import argparse
    import sys
    import astropy.table
    from pathlib import Path
    import astropy.units as u
    from ..io.serialize import read

    parser = argparse.ArgumentParser(prog=__name__)
    parser.add_argument("images", type=Path)
    parser.add_argument('output', nargs="?", type=argparse.FileType("w"), default=sys.stdout)

    args = parser.parse_args()

    images = read(args.images)
    psfs = []
    for visit, detector in images:
        bbox = images[visit, detector].reader.readBBox()
        psf = images[visit, detector].reader.readPsf()
        psfs.append(
            dict(
                visit=visit,
                detector=detector,
                psf=psf.computeShape(bbox.getCenter()).getDeterminantRadius() * u.pixel * (0.263 * u.arcsec / u.pixel)
            )
        )
        
    astropy.table.Table(psfs).write(args.output, format="ascii.ecsv")

if __name__ == "__main__":
    main()
