import logging
import parsl
from .parsl_config import load_parsl
from .tasks import (
    get_fakes,
    filter_fakes,
    get_images,
    detection,
    shuffle,
    search,
    gather,
    refine,
    match_fakes_catalog,
    match_fakes_clusters_line,
    match_fakes_clusters_points,
    fakes_info,
    summary,
    launch_for_outputs
)

logging.basicConfig()
log = logging.getLogger(__name__)

def main():
    import argparse
    import sys
    from pathlib import Path
    from parsl import File

    parser = argparse.ArgumentParser(prog=__name__)
    parser.add_argument("label", type=str)
    parser.add_argument("--velocity", required=True, nargs=2, type=float)
    parser.add_argument("--angle", required=True, nargs=2, type=float)
    parser.add_argument("--threshold", required=True, type=int)
    parser.add_argument("--threshold-type", required=True, type=str)
    parser.add_argument("--snrs", nargs="+", type=float, default=[5.0, 4.5, 4.0, 3.5, 3.0])
    parser.add_argument("--dx", default=None, type=float)
    parser.add_argument("--detectors", nargs="+", type=int, default=list(range(1, 63)))
    parser.add_argument("--exclude-detectors", nargs="+", type=int, default=[2, 61])

    parser.add_argument("--processes", type=int, default=12)
    
    args = parser.parse_args()

    dx_lookup = {
        0: 1,
        1: 1,
        2: 2,
        3: 3,
        4: 10,
        5: 10,
    }

    work_dir = Path("search/tno_search")
    collections = ["DEEP/20190403/A0c"]
    detectors = list(filter(lambda x : x not in args.exclude_detectors, args.detectors))
    # snrs = list(range(3, 6))
    snrs = list(args.snrs)
    
    futures = []
    config, dfk = load_parsl(args.processes)

    def launch_tasks():
        # get fakes
        future, all_fakes = launch_for_outputs(
            get_fakes, 
            outputs=[File(work_dir / "fakes.pkl")],
            work_dir=work_dir,
        )
        futures.append(future)
        for collection in collections:
            c = work_dir / collection
            for detector in detectors:
                for_summary = []
                d = c / f"detector_{detector}"
                # filter fakes
                future, fakes = launch_for_outputs(
                    filter_fakes,
                    collection, "differenceExp", detector,
                    inputs=all_fakes,
                    outputs=[File(d / "fakes.pkl")],
                    work_dir=d,
                )
                # get images            
                future, images = launch_for_outputs(
                    get_images,
                    collection, "differenceExp", detector,
                    outputs=[File(d / "images.pkl")],
                    work_dir=d,
                )
                futures.append(future)
                for snr in snrs:
                    s = d / f"snr_{float(snr)}"
                    # detection
                    future, catalog = launch_for_outputs(
                        detection,
                        snr,
                        no_masks=False,
                        inputs=images,
                        outputs=[File(s / "regular" / "catalog.pkl")],
                        work_dir=s,
                    )
                    futures.append(future)

                    future, catalog_matches = launch_for_outputs(
                        match_fakes_catalog,
                        fakes[0],
                        catalog[0],
                        inputs=fakes + catalog,
                        outputs=[File(s / "regular" / "catalog_matches.pkl")],
                        work_dir=s,
                    )
                    futures.append(future)

                    future, catalog_no_masks = launch_for_outputs(
                        detection,
                        snr, no_masks=True,
                        inputs=images,
                        outputs=[File(s / "regular" / "catalog_no_masks.pkl")],
                        work_dir=s,
                    )
                    futures.append(future)

                    future, catalog_no_masks_matches = launch_for_outputs(
                        match_fakes_catalog,
                        fakes[0], 
                        catalog_no_masks[0],
                        inputs=fakes + catalog_no_masks,
                        outputs=[File(s / "regular" / "catalog_no_masks_matches.pkl")],
                        work_dir=s,
                    )
                    futures.append(future)

                    # shuffle
                    future, shuffled_catalog = launch_for_outputs(
                        shuffle,
                        inputs=catalog,
                        outputs=[File(s / "shuffled" / "catalog.pkl")],
                        work_dir=s,
                    )
                    futures.append(future)

                    future, shuffled_matches = launch_for_outputs(
                        match_fakes_catalog,
                        fakes[0], 
                        shuffled_catalog[0],
                        inputs=fakes + shuffled_catalog,
                        outputs=[File(s / "shuffled" / "catalog.pkl")],
                        work_dir=s,
                    )
                    futures.append(future)

                    for time_type in ["regular", "shuffled"]:
                        r = s / time_type / args.label
                        # search
                        if args.dx:
                            dx = args.dx
                        else:
                            dx = dx_lookup[snr]
                        future, clusters = launch_for_outputs(
                            search,
                            images[0],
                            catalog[0],
                            dx,
                            args.velocity,
                            args.angle,
                            args.threshold,
                            args.threshold_type,
                            inputs=images + catalog,
                            outputs=[File(r / "clusters.pkl"), File(r / "gathered.pkl")],
                            work_dir=r,
                        )
                        futures.append(future)
                        for_summary.extend(clusters)

                        future, clusters_line_matches = launch_for_outputs(
                            match_fakes_clusters_line,
                            fakes[0], 
                            clusters[0],
                            inputs=fakes + clusters,
                            outputs=[File(r / "clusters_match_line.pkl")],
                            work_dir=r,
                        )
                        futures.append(future)

                        future, gathered_line_matches = launch_for_outputs(
                            match_fakes_clusters_line,
                            fakes[0], 
                            clusters[1],
                            inputs=fakes + clusters,
                            outputs=[File(r / "gathered_match_line.pkl")],
                            work_dir=r,
                        )
                        futures.append(future)

                        future, clusters_points_matches = launch_for_outputs(
                            match_fakes_clusters_points,
                            fakes[0], 
                            clusters[0],
                            inputs=fakes + clusters,
                            outputs=[File(r / "clusters_match_points.pkl")],
                            work_dir=r,
                        )
                        futures.append(future)
                        
                        future, gathered_points_matches = launch_for_outputs(
                            match_fakes_clusters_points,
                            fakes[0], 
                            clusters[1],
                            inputs=fakes + clusters,
                            outputs=[File(r / "gathered_match_points.pkl")],
                            work_dir=r,
                        )
                        futures.append(future)
                future, summaries = launch_for_outputs(
                    summary,
                    50, 50,
                    inputs=images + for_summary,
                    outputs=[File(Path(s.filepath).parent / Path(s.filepath).name.replace(".pkl", "_summary.pkl")) for s in for_summary],
                    work_dir=d,
                )
                futures.append(future)

    try:
        launch_tasks()
        for future in futures:
            if future is not None:
                future.exception()
    except Exception as e:
        raise e
    finally:
        dfk.cleanup()
        parsl.DataFlowKernelLoader.clear()
