from .fitting import logL_position

def forced_exposures(exposures, points):
    import lsst.geom
    import astropy.table
    
    results = []
    for exposure, row in zip(exposures, points):
        result = logL_position(exposure, lsst.geom.Point2D(row['i_x'], row['i_y']), [0, 0])
        result = {
            f"forced_{key}": value
            for key, value in result.items()
        }
        result['forced_i_x'] = row['i_x']
        result['forced_i_y'] = row['i_y']
        result['forced_exposure'] = exposure.getInfo().getVisitInfo().getId()
        result['forced_detector'] = exposure.getDetector().getId()
        result['forced_time'] = exposure.getInfo().getVisitInfo().date.toAstropy()
        results.append(result)

    return astropy.table.Table(results)
