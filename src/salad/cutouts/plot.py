from ..serialize import read, write
import logging

logging.basicConfig()
log = logging.getLogger(__name__)

def plot_cluster_cutouts(
    cluster, 
    cutouts=None,
    cols=None, rows=None, 
    component="", 
    title="{visit}\n({x}, {y})\nSNR={significance:0.1f}", 
    show_colorbar=False, 
    highlight_points=True,
    limit=None,
    stretch="linear",
    scale="zscale",
    **kwargs
):
    import matplotlib.pyplot as plt
    import lsst.afw.display as afwDisplay
    import numpy as np
    
    if "forced_line" in cluster.extra:
        catalog = cluster.extra['forced_line']
        cutouts = np.array(cluster.extra['cutouts'])
        get_significance = lambda row : row['forced_SNR']
        exposure_col = "forced_expnum"
    elif "forced_points" in cluster.extra:
        catalog = cluster.extra['forced_points']
        if cutouts is None:
            raise Exception("must pass cutouts if plotting forced_points cluster")

        get_significance = lambda row : row['forced_SNR']      
        exposure_col = "forced_expnum"      
    else:
        catalog = cluster.extra['join']
        cutouts = cluster.extra['cutouts']
        get_significance = lambda row : row['significance']
        exposure_col = "expnum"
    
    idx = np.argsort(catalog[exposure_col])
    catalog = catalog[idx]
    cutouts = np.array(cutouts)[idx]
    
    if limit:
        cutouts = cutouts[:limit]
        catalog = catalog[:limit]
        
    included_exposures = np.unique(cluster.extra['join']['expnum'])
    
    afwDisplay.setDefaultBackend("matplotlib")
    if rows is not None and cols is None:
        cols = len(cutouts) / rows
    elif rows is None and cols is not None:
        rows = len(cutouts) / cols
    elif rows is None and cols is None:
        cols = round((len(cutouts))**0.5 + 0.5)
        rows = len(cutouts)/cols
                
    if cols > len(cutouts):
        cols = len(cutouts)
        rows = 1

    if (cols - int(cols)) != 0:
        cols += 1
    cols = int(cols)
        
    if (rows - int(rows)) != 0:
        rows += 1
    rows = int(rows)
        
    fig = plt.figure(figsize=(cols*2, rows*2), **kwargs)
    axs = fig.subplots(rows, cols)
    axs = np.atleast_2d(axs)
    display = afwDisplay.Display(frame=fig)
    display.scale(stretch, scale)
    
    for ax, row, cutout in zip(axs.flatten(), catalog, cutouts):
        plt.sca(ax)
        a = cutout
        if component:
            a = getattr(cutout, component)
        display.mtv(a)

        visit = cutout.getInfo().getVisitInfo().getId()
        x = int(row['i_x'])
        y = int(row['i_y'])
        significance = get_significance(row)
        plt.title(title.format(**locals()))

        if highlight_points:
            if np.isin(visit, included_exposures):
                ax.spines['bottom'].set_color('red')
                ax.spines['top'].set_color('red')
                ax.spines['left'].set_color('red')
                ax.spines['right'].set_color('red')

        display.show_colorbar(show=show_colorbar)
        
    return fig, axs



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
    parser.add_argument("--cutouts", default=None)
    parser.add_argument("--output-format", default="pkl")
    parser.add_argument("--rows", default=None, type=int)
    parser.add_argument("--cols", default=None, type=int)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--no-axes", action="store_true")

    args = parser.parse_args()

    kwargs = vars(args)
    o = kwargs.pop("output")
    i = kwargs.pop("input")
    no_axes = kwargs.pop("no_axes")
    output_format = kwargs.pop("output_format")

    cluster = read(i)
    if args.cutouts:
        kwargs['cutouts'] = read(args.cutouts).extra['cutouts']
    fig, axs = plot_cluster_cutouts(cluster, **kwargs)
    if no_axes:
        for ax in axs.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

    if output_format == "pkl":
        write((fig, axs), o)
    else:
        fig.savefig(o, format=output_format, bbox_inches='tight')


if __name__ == "__main__":
    main()
