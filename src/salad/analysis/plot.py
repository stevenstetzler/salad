import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import numpy as np
import astropy.units as u
from ..serialize import read
import logging

logging.basicConfig()
log = logging.getLogger(__name__)

def plot_cluster(cluster, **kwargs):
    p = cluster.points
    x, y = p[:, 2][:, None], p[:, :2]
    coord = (SkyCoord(y[:, 0] * u.deg, y[:, 1] * u.deg))
    plt.scatter(coord.ra, coord.dec, **kwargs)

def plot_line(line, x, **kwargs):
    y = line.predict(x)
    coord = (SkyCoord(y[:, 0], y[:, 1]))
    plt.plot(coord.ra, coord.dec, **kwargs)


def plot_result(result, x, **kwargs):
    _x = np.linspace(x.min(), x.max(), 100)[:, None]
    if hasattr(result.alpha, "unit"):
        _x *= result.alpha.unit / result.beta.unit

    _y = _x @ result.beta + result.alpha
    coord = (SkyCoord(_y[:, 0], _y[:, 1]))
    plt.plot(coord.ra, coord.dec, **kwargs)
    
def plot_fake(orbit, fakes, **kwargs):
    r = fakes[fakes['ORBITID'] == orbit]
    
    coord = (SkyCoord(r['RA'] * u.deg, r['DEC'] * u.deg))
    plt.scatter(coord.ra, coord.dec, **kwargs)

def plot_catalog(catalog, **kwargs):
    coord = (SkyCoord(catalog.ra, catalog.dec))
    plt.scatter(coord.ra, coord.dec, **kwargs)

def plot_summary_coadds(s):
    fig = plt.figure(dpi=150)
    axs = fig.subplots(1, 5)
    plt.sca(axs[0])
    plt.imshow(s['coadd']['mean'], cmap='gray_r')
    plt.title("mean")
    plt.xticks([])
    plt.yticks([])

    plt.sca(axs[1])
    plt.imshow(s['coadd']['sum'], cmap='gray_r')
    plt.title("sum")
    plt.xticks([])
    plt.yticks([])

    plt.sca(axs[2])
    plt.imshow(s['coadd']['median'], cmap='gray_r')
    plt.title("median")
    plt.xticks([])
    plt.yticks([])

    plt.sca(axs[3])
    plt.imshow(s['coadd']['weighted'], cmap='gray_r')
    plt.title("weighted")
    plt.xticks([])
    plt.yticks([])
    
    plt.sca(axs[4])
    plt.imshow(np.bitwise_or.reduce(s['mask'], axis=0))
    plt.title("mask")
    plt.xticks([])
    plt.yticks([])

    return fig

def plot_summary_light_curve(s):
    fig = plt.figure(dpi=150)
    axs = fig.subplots(3, 1, sharex=True)
    plt.sca(axs[0])
    plt.errorbar(
        np.arange(len(s['light_curve']['flux'])), 
        s['light_curve']['flux'], 
        yerr=s['light_curve']['sigma'],
        fmt='o',
    #     c=s['light_curve']['mask'],
        ms=2,
        lw=1,
    )
    plt.title("Flux (zp=31)")
    plt.sca(axs[1])
    plt.scatter(
        np.arange(len(s['light_curve']['flux'])), 
        s['light_curve']['mag'],
        c=s['light_curve']['mask'] != 0,
        s=2
    )
    plt.title("Mag")
    plt.sca(axs[2])
    plt.scatter(
        np.arange(len(s['light_curve']['flux'])), 
        s['light_curve']['snr'],
        c=s['light_curve']['mask'] != 0,
        s=2
    )
    plt.title("SNR")
    return fig


def plot_cluster_cutouts(
    cluster, 
    cols=None, rows=None, 
    component="", 
    title="{visit}",#"\n({x}, {y})\nSNR={significance:0.1f}", 
    show_colorbar=False, 
    highlight_points=True,
    limit=None,
    only_points=False,
    stretch="linear",
    scale="zscale",
    **kwargs
):
    import matplotlib.pyplot as plt
    import lsst.afw.display as afwDisplay
    import lsst.geom
    import numpy as np
    
    expnum = sorted(list(cluster.cutouts.keys()))
    
    if only_points:
        expnum = sorted(list(set(cluster.points[:, 3])))
    
    if limit:
        expnum = expnum[:limit]
    
    if len(expnum) == 0:
        return None, None
    
    afwDisplay.setDefaultBackend("matplotlib")
    if rows is not None and cols is None:
        cols = len(expnum) / rows
    elif rows is None and cols is not None:
        rows = len(expnum) / cols
    elif rows is None and cols is None:
        cols = round((len(expnum))**0.5 + 0.5)
        rows = len(expnum)/cols
                
    if cols > len(expnum):
        cols = len(expnum)
        rows = 1

    if (cols - int(cols)) != 0:
        cols += 1
    cols = int(cols)
        
    if (rows - int(rows)) != 0:
        rows += 1
    rows = int(rows)

    print(rows, cols)
    fig = plt.figure(figsize=(cols*2, rows*2), **kwargs)
    axs = fig.subplots(rows, cols)
    axs = np.atleast_2d(axs)
    display = afwDisplay.Display(frame=fig)
    display.scale(stretch, scale)
    
    for ax, e in zip(axs.flatten(), expnum):
        plt.sca(ax)
        cutout = cluster.cutouts[e]
        center = cluster.centers[e]
        a = cutout
        if component:
            a = getattr(cutout, component)
        display.mtv(a)

        visit = cutout.getInfo().getVisitInfo().getId()
#         x = center.getX()
#         y = center.getY()
#         snr = 
#         x = int(row['i_x'])
#         y = int(row['i_y'])
#         significance = get_significance(row)
        plt.title(title.format(**locals()))

        if highlight_points:
            if np.isin(visit, cluster.points[:, 3]):
                mask = cluster.points[:, 3] == visit
                ax.spines['bottom'].set_color('red')
                ax.spines['top'].set_color('red')
                ax.spines['left'].set_color('red')
                ax.spines['right'].set_color('red')
                for ra, dec, _, _ in cluster.points[mask]:
                    p = cutout.wcs.skyToPixel(lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees))
                    display.dot("+", p.getX(), p.getY())

        display.show_colorbar(show=show_colorbar)
        
    if len(axs.flatten()) > len(expnum):
        for ax in axs.flatten()[len(expnum):]:
            ax.remove()
    return fig, axs

def summary_coadds_plot(cluster, exclude_masks=["DETECTED", "DETECTED_NEGATIVE", "FAKE", "INJECTED", "INJECTED_TEMPLATE", "NOT_DEBLENDED", "STREAK"]):
    s = cluster.summary
    points = cluster.points
    
    included = []
    excluded = []
    e2 = points[:, -1].astype(int)
    for i, e in enumerate(s['expnum']):
        if e in e2:
            included.append(i)
        else:
            excluded.append(i)

    included = np.array(included)
    excluded = np.array(excluded)
    total = np.array([i for i in range(len(s['expnum']))])
    
    # coadds plot:
    fig = plt.figure(dpi=150)
    axs = fig.subplots(3, 5, sharex=True, sharey=True)
    for i, idx in enumerate([total, included, excluded]):
        for j, value in enumerate(['mean', 'sum', 'median', 'weighted', 'mask']):
            if len(idx) == 0:
                continue
            image = s['image'][idx]
            variance = s['variance'][idx]
            mask = s['mask'][idx]
            
            plt.sca(axs[i, j])
            cmap = 'gray_r'
            if value == 'weighted':
                v = (np.sum((image / variance), axis=0) / np.sum(1/variance, axis=0))
            elif value == 'mask':
                v = np.bitwise_or.reduce(mask, axis=0)
                for m in exclude_masks:
                    b = s['mask_plane_dict'][0][m]
                    v[np.where(((v >> b) & 1) == 1)] -= 2**b
                cmap = None
            else:
                v = getattr(np, value)(image, axis=0)
                
                
            plt.imshow(v, cmap=cmap)
            plt.title(value)
            if j == 0:
                label = {
                    0: f"all (N={len(total)})",
                    1: f"detected (N={len(included)})",
                    2: f"undetected (N={len(excluded)})"
                }
                plt.ylabel(label.get(i))
            plt.xticks([])
            plt.yticks([])
    return fig

import matplotlib
from matplotlib.lines import Line2D

def summary_lightcurve_plot(cluster, exclude_masks=["DETECTED", "DETECTED_NEGATIVE", "FAKE", "INJECTED", "INJECTED_TEMPLATE", "NOT_DEBLENDED", "STREAK"]):
    # flux / mag / snr
    # colors: included / mask
    s = cluster.summary
    points = cluster.points
    
    included = []
    excluded = []
    e2 = points[:, -1].astype(int)
    for i, e in enumerate(s['expnum']):
        included.append(e in e2)
        excluded.append(e not in e2)

    included = np.array(included)
    excluded = np.array(excluded)
    total = np.array([i for i in range(len(s['expnum']))])
    
    v = s['light_curve']['mask']
    for m in exclude_masks:
        b = s['mask_plane_dict'][0][m]
        v[np.where(((v >> b) & 1) == 1)] -= 2**b    

    included_mask = v == 0
    excluded_mask = v != 0
    
    # coadds plot:
    fig = plt.figure(dpi=150, figsize=(9, 5))
    axs = fig.subplots(3, 2, sharex=True, sharey=False)
    axs = np.atleast_2d(axs)
    x = np.arange(len(s['expnum']))
    
    for j, (incl, excl) in enumerate(zip([included, included_mask], [excluded, excluded_mask])):
        plt.sca(axs[0, j])
        plt.errorbar(
            x[incl], 
            s['light_curve']['flux'][incl], 
            yerr=s['light_curve']['sigma'][incl],
            fmt='o',
            ms=2,
        )
        if excl.sum() > 0:
            plt.errorbar(
                x[excl], 
                s['light_curve']['flux'][excl], 
                yerr=s['light_curve']['sigma'][excl],
                fmt='o',
                ms=2,
            )

        plt.sca(axs[1, j])
        plt.scatter(
            x[incl], 
            s['light_curve']['mag'][incl], 
            s=2,
        )
        if len(excl) > 0:
            plt.scatter(
                x[excl], 
                s['light_curve']['mag'][excl], 
                s=2,
            )

        plt.gca().invert_yaxis()

        plt.sca(axs[2, j])
        plt.scatter(
            x[incl], 
            s['light_curve']['snr'][incl], 
            s=2,
        )
        if len(excl) > 0:
            plt.scatter(
                x[excl], 
                s['light_curve']['snr'][excl], 
                s=2,
            )
        
    plt.sca(axs[0, 0])
    plt.ylabel("Flux")
    plt.sca(axs[1, 0])
    plt.ylabel("Mag")
    plt.sca(axs[2, 0])
    plt.ylabel("SNR")
    plt.xlabel("Obs #")
    custom_lines = [
        Line2D([0], [0], marker="o", lw=0, color='C0', markersize=2, markerfacecolor="C0", label=f"Detection"),
        Line2D([0], [0], marker="o", lw=0, color='C1', markersize=2, markerfacecolor="C1", label=f"Non-detection"),        
    ]
    fig.legend(handles=custom_lines, loc='upper left')
    
    plt.sca(axs[2, 1])
    plt.xlabel("observation number")
    custom_lines = [
        Line2D([0], [0], marker="o", lw=0, color='C0', markersize=2, markerfacecolor="C0", label=f"Unmasked"),
        Line2D([0], [0], marker="o", lw=0, color='C1', markersize=2, markerfacecolor="C1", label=f"Masked"),        
    ]
    fig.legend(handles=custom_lines, loc='upper right')
    
    return fig

def summary_cutouts(
    cluster, 
    cols=None, rows=None, 
    component="image", 
    title="{e}",#"\n({x}, {y})\nSNR={significance:0.1f}", 
    show_colorbar=False, 
    highlight_points=True,
    limit=None,
    only_points=False,
    stretch="linear",
    scale="zscale",
    share_colorbar=False,
    **kwargs
):
    import matplotlib.pyplot as plt
    import numpy as np
    
    s = cluster.summary
    expnum = s['expnum']
        
    if only_points:
        expnum = sorted(list(set(cluster.points[:, 3])))
    
    if limit:
        expnum = expnum[:limit]
    
    if len(expnum) == 0:
        return None, None
    
    if rows is not None and cols is None:
        cols = len(expnum) / rows
    elif rows is None and cols is not None:
        rows = len(expnum) / cols
    elif rows is None and cols is None:
        cols = round((len(expnum))**0.5 + 0.5)
        rows = len(expnum)/cols
                
    if cols > len(expnum):
        cols = len(expnum)
        rows = 1

    if (cols - int(cols)) != 0:
        cols += 1
    cols = int(cols)
        
    if (rows - int(rows)) != 0:
        rows += 1
    rows = int(rows)

    print(rows, cols)
    fig = plt.figure(figsize=(cols*2, rows*2), **kwargs)
    axs = fig.subplots(rows, cols)
    axs = np.atleast_2d(axs)
    
    if share_colorbar:
        vmin = cluster.summary[component].min()
        vmax = cluster.summary[component].max()
    else:
        vmin = None
        vmax = None
        
    for ax, e in zip(axs.flatten(), expnum):
        plt.sca(ax)
        center = cluster.centers[e]
        cutout = cluster.summary[component][s['expnum'] == e][0]

        plt.imshow(cutout, cmap="gray_r", vmin=vmin, vmax=vmax)
        plt.title(title.format(**locals()))

        if highlight_points:
            if np.isin(e, cluster.points[:, 3]):
                mask = cluster.points[:, 3] == e
                ax.spines['bottom'].set_color('red')
                ax.spines['top'].set_color('red')
                ax.spines['left'].set_color('red')
                ax.spines['right'].set_color('red')

    if len(axs.flatten()) > len(expnum):
        for ax in axs.flatten()[len(expnum):]:
            ax.remove()
    return fig


def main():
    import argparse
    import sys
    import os

    parser = argparse.ArgumentParser(prog=__name__)

    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--plot-type", default="summary_coadds_plot", type=str)
    args = parser.parse_args()

    clusters = read(args.input)
    for k in clusters:
        cluster = clusters[k]
        if args.plot_type == "cutout":
            for expnum, image in zip(cluster.summary['expnum'], cluster.summary['image']):
                p = os.path.join(args.output, f"cluster_{k}", f"cutout_{expnum}.png")
                if os.path.exists(p):
                    log.info("skipping existing %s", p)
                    continue
                os.makedirs(os.path.dirname(p), exist_ok=True)
                fig = plt.figure()
                plt.imshow(image, cmap='gray_r')
                log.info("saving %s to %s", args.plot_type, p)
                fig.suptitle(f"exposure {expnum}")
                fig.tight_layout()
                fig.savefig(p)
                plt.close()
                del fig
        else:
            fig = globals()[args.plot_type](cluster)
            p = os.path.join(args.output, f"cluster_{k}", f"{args.plot_type}.png")
            if os.path.exists(p):
                log.info("skipping existing %s", p)
                continue     
            os.makedirs(os.path.dirname(p), exist_ok=True)
            
            log.info("saving %s to %s", args.plot_type, p)
            fig.suptitle(f"cluster {k}")
            fig.tight_layout()
            fig.savefig(p)
            plt.close()
            del fig
