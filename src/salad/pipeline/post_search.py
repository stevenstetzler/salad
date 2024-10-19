import logging
import parsl
from parsl import bash_app, File
import os
from pathlib import Path
from .parsl_config import load_parsl
from .tasks import (
    launch_for_outputs, 
    detection, project, 
    cluster_hough, find_clusters, 
    refine, gather, filter_clusters, join, split_clusters,
    recover, detectable_in_catalog, detectable_in_search,
    filter_fakes, get_images, get_fakes, shuffle, summary, deduplicate, plot
)

logging.basicConfig()
log = logging.getLogger(__name__)

def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser(prog=__name__)
    parser.add_argument("work_dir")
    parser.add_argument("label")
    parser.add_argument("--collections", nargs="+", default=["DEEP/20190403/A0c"])
    parser.add_argument("--detectors", nargs="+", type=int, default=list(range(1, 63)))
    parser.add_argument("--snrs", nargs="+", type=float, default=[5.0, 4.5, 4.0, 3.5, 3.0])
    parser.add_argument("--cutout-width", type=int, default=50)
    parser.add_argument("--cutout-height", type=int, default=50)
    parser.add_argument("--summary-step", default="filtered")
    parser.add_argument("--processes", "-J", type=int, default=1)

    args = parser.parse_args()

    def per_snr(work_dir, fakes, collection, detector, snr):
        summaries_for = [File(f) for f in Path(work_dir).rglob(f"{args.summary_step}.pkl")]
        return summaries_for

    def per_detector(work_dir, fakes, collection, detector):
        os.makedirs(work_dir, exist_ok=True)
        images = File(os.path.join(work_dir, "images.pkl"))

        summaries_for = []
        for snr in args.snrs:
            summaries_for.extend(per_snr(os.path.join(work_dir, f"snr_{snr}"), fakes, collection, detector, snr))
        
        summaries = []
        for f in summaries_for:
            p = os.path.join(
                os.path.dirname(f.filepath), os.path.basename(f.filepath).replace(".pkl", "_summary.pkl")
            )
            summaries.append(File(p))
        
        future, summaries = launch_for_outputs(
            summary,
            args.cutout_width, args.cutout_height,
            inputs=[images] + summaries_for,
            outputs=summaries,
            work_dir=work_dir,
        )
        futures.append(future)

        for s in summaries:
            outputs = File(os.path.join(os.path.dirname(s.filepath), "summary_plots"))
            future = bash_app(plot)("summary_cutouts", inputs=[s], outputs=[outputs])
            futures.append(future)
            future = bash_app(plot)("summary_coadds_plot", inputs=[s], outputs=[outputs])
            futures.append(future)
            future = bash_app(plot)("summary_lightcurve_plot", inputs=[s], outputs=[outputs])
            futures.append(future)

    def per_collection(work_dir, fakes, collection):
        os.makedirs(work_dir, exist_ok=True)
        for detector in args.detectors:
            per_detector(os.path.join(work_dir, f"detector_{detector}"), fakes, collection, detector)

    def launch_tasks(work_dir):
        os.makedirs(work_dir, exist_ok=True)
        fakes = File(os.path.join(work_dir, "fakes.pkl"))
        for collection in args.collections:
            per_collection(os.path.join(work_dir, collection), fakes, collection)
    
    config, dfk = load_parsl(args.processes)
    
    futures = []
    try:
        launch_tasks(args.work_dir)
        for future in futures:
            if future is not None:
                future.exception()
    except Exception as e:
        raise e
    finally:
        dfk.cleanup()
        parsl.DataFlowKernelLoader.clear()

if __name__ == "__main__":
    main()
