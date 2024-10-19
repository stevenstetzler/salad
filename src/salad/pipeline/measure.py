import logging
import parsl
from parsl import bash_app, File
import os
from .parsl_config import load_parsl
from .tasks import (
    launch_for_outputs, 
    get_cutouts,
    forced_points, forced_line,
    coadd_points, coadd_line,
    fit_line, plot_cutouts
)

logging.basicConfig()
log = logging.getLogger(__name__)


def main():
    import argparse
    import sys
    from pathlib import Path
    import re

    parser = argparse.ArgumentParser(prog=__name__)
    parser.add_argument("work_dir")
    parser.add_argument("--clusters-dir", default="refined_clusters_2_filtered_joined_split")
    parser.add_argument("--collections", nargs="+", default=["DEEP/20190403/A0c"])
    parser.add_argument("--detectors", nargs="+", type=int, default=list(range(1, 63)))
    parser.add_argument("--snrs", nargs="+", type=float, default=[5, 4, 3])
    parser.add_argument("--datasetType", default="differenceExp")
    # parser.add_argument("--velocity", nargs=2, type=float, default=[0.1, 0.5])
    # parser.add_argument("--angle", nargs=1, type=float, default=[120, 240])
    # parser.add_argument("--dx", type=float, default=10)
    # parser.add_argument("--vote-threshold", type=int, default=25)
    # parser.add_argument("--min-points", type=int, default=15)
    # parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--processes", "-J", type=int, default=1)

    args = parser.parse_args()

    config, dfk = load_parsl(args.processes)
    
    futures = []

    def per_cluster(clusters_dir, cluster_path, collection, detector, snr, images):
        if re.compile(r"\d+").match(cluster_path.parent.name):
            cluster = File(str(cluster_path))
            cutouts_path = cluster_path.parent / "cutouts.pkl" 
            cutouts = File(str(cutouts_path))
            future, cutouts = launch_for_outputs(
                get_cutouts,
                collection, args.datasetType, detector,
                inputs=[cluster],
                outputs=[cutouts],
                stdout=cluster_path.parent / "cutouts.stdout",
                stderr=cluster_path.parent / "cutouts.stderr",
            )
            futures.append(future)

            # plot_cutouts_output = File(str(cluster_path.parent / cluster_path.name.replace(".pkl", "_plot_cutouts.pkl")))
            # future, plot_cutouts_output = launch_for_outputs(
            #     plot_cutouts,
            #     inputs=cutouts,
            #     outputs=[plot_cutouts_output],
            #     stdout=cluster_path.parent / cluster_path.name.replace(".pkl", "_plot_cutouts.stdout"),
            #     stderr=cluster_path.parent / cluster_path.name.replace(".pkl", "_plot_cutouts.stderr"),
            # )
            # futures.append(future)

            plot_cutouts_output_png = File(str(cluster_path.parent / "plot_cutouts.png"))
            future, plot_cutouts_output_png = launch_for_outputs(
                plot_cutouts,
                output_format="png",
                inputs=cutouts,
                outputs=[plot_cutouts_output_png],
                stdout=cluster_path.parent / "plot_cutouts_png.stdout",
                stderr=cluster_path.parent / "plot_cutouts_png.stderr",
            )
            futures.append(future)


            forced_points_path = cluster_path.parent / "forced_points.pkl"
            forced_points_output = File(str(forced_points_path))
            future, forced_points_output = launch_for_outputs(
                forced_points,
                inputs=cutouts,
                outputs=[forced_points_output],
                stdout=cluster_path.parent / "forced_points.stdout",
                stderr=cluster_path.parent / "forced_points.stderr",
            )
            futures.append(future)

            coadd_points_path = cluster_path.parent / "coadd_points.pkl"
            coadd_points_output = File(str(coadd_points_path))
            future, coadd_points_output = launch_for_outputs(
                coadd_points,
                inputs=cutouts,
                outputs=[coadd_points_output],
                stdout=cluster_path.parent / "coadd_points.stdout",
                stderr=cluster_path.parent / "coadd_points.stderr",
            )
            futures.append(future)

            forced_line_path = cluster_path.parent / "forced_line.pkl"
            forced_line_output = File(str(forced_line_path))
            future, forced_line_output = launch_for_outputs(
                forced_line,
                images,
                inputs=cutouts,
                outputs=[forced_line_output],
                stdout=cluster_path.parent / "forced_line.stdout",
                stderr=cluster_path.parent / "forced_line.stderr",
            )
            futures.append(future)

            # plot_line_output = File(str(cluster_path.parent / cluster_path.name.replace(".pkl", "_plot_line.pkl")))
            # future, plot_line_output = launch_for_outputs(
            #     plot_cutouts,
            #     inputs=forced_line_output,
            #     outputs=[plot_line_output],
            #     stdout=cluster_path.parent / cluster_path.name.replace(".pkl", "_plot_line.stdout"),
            #     stderr=cluster_path.parent / cluster_path.name.replace(".pkl", "_plot_line.stderr"),
            # )
            # futures.append(future)

            plot_line_output_png = File(str(cluster_path.parent / "plot_line.png"))
            future, plot_line_output_png = launch_for_outputs(
                plot_cutouts,
                output_format="png",
                inputs=forced_line_output,
                outputs=[plot_line_output_png],
                stdout=cluster_path.parent / "plot_line_png.stdout",
                stderr=cluster_path.parent / "plot_line_png.stderr",
            )
            futures.append(future)

            coadd_line_path = cluster_path.parent / "coadd_line.pkl"
            coadd_line_output = File(str(coadd_line_path))
            future, coadd_line_output = launch_for_outputs(
                coadd_line,
                images,
                inputs=cutouts,
                outputs=[coadd_line_output],
                stdout=cluster_path.parent / "coadd_line.stdout",
                stderr=cluster_path.parent / "coadd_line.stderr",
            )
            futures.append(future)

            fit_line_path = cluster_path.parent / "fit_line.pkl"
            fit_line_output = File(str(fit_line_path))
            future, fit_line_output = launch_for_outputs(
                fit_line,
                images,
                inputs=cutouts,
                outputs=[fit_line_output],
                stdout=cluster_path.parent / "fit_line.stdout",
                stderr=cluster_path.parent / "fit_line.stderr",
            )
            futures.append(future)

            coadd_fit_line_path = cluster_path.parent / "coadd_fit_line.pkl"
            coadd_fit_line_output = File(str(coadd_fit_line_path))
            future, fit_line_output = launch_for_outputs(
                coadd_line,
                images,
                inputs=fit_line_output,
                outputs=[coadd_fit_line_output],
                stdout=cluster_path.parent / "coadd_fit_line.stdout",
                stderr=cluster_path.parent / "coadd_fit_line.stderr",
            )
            futures.append(future)

    def per_catalog(work_dir, collection, detector, snr, images):
        os.makedirs(work_dir, exist_ok=True)
        clusters_dir = f"{work_dir}/{args.clusters_dir}"
        print(clusters_dir)
        for cluster_path in Path(clusters_dir).rglob("cluster.pkl"):
            per_cluster(clusters_dir, cluster_path, collection, detector, snr, images)

    def per_snr(work_dir, collection, detector, snr, images):
        os.makedirs(work_dir, exist_ok=True)
        for subdir in ["unshuffled", "shuffled"]:
            per_catalog(os.path.join(work_dir, subdir), collection, detector, snr, images)

    def per_detector(work_dir, collection, detector):
        os.makedirs(work_dir, exist_ok=True)
        images = File(str(os.path.join(work_dir, "images.pkl")))
        for snr in args.snrs:
            per_snr(os.path.join(work_dir, f"snr_{snr}"), collection, detector, snr, images)

    def per_collection(work_dir, collection):
        os.makedirs(work_dir, exist_ok=True)
        for detector in args.detectors:
            per_detector(os.path.join(work_dir, f"detector_{detector}"), collection, detector)

    def launch_tasks(work_dir):
        os.makedirs(work_dir, exist_ok=True)
        for collection in args.collections:
            per_collection(os.path.join(work_dir, collection), collection)

    launch_tasks(args.work_dir)
    for future in futures:
        if future is not None:
            future.exception()

    dfk.cleanup()
    parsl.DataFlowKernelLoader.clear()

if __name__ == "__main__":
    main()
