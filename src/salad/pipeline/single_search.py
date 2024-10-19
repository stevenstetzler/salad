import logging
import parsl
from parsl import bash_app, File
import os
from .parsl_config import load_parsl
from .tasks import (
    launch_for_outputs, 
    detection,
    single_search,
    filter_fakes, get_images, get_fakes, shuffle
)

logging.basicConfig()
log = logging.getLogger(__name__)

def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser(prog=__name__)
    parser.add_argument("work_dir")
    parser.add_argument("--collections", nargs="+", default=["DEEP/20190403/A0c"])
    parser.add_argument("--detectors", nargs="+", type=int, default=list(range(1, 63)))
    parser.add_argument("--snrs", nargs="+", type=float, default=[5.0, 4.5, 4.0, 3.5, 3.0])
    parser.add_argument("--datasetType", default="differenceExp")
    parser.add_argument("--velocity", nargs=2, type=float, default=[0.1, 0.5])
    parser.add_argument("--angle", nargs=1, type=float, default=[120, 240])
    parser.add_argument("--dx", type=float, default=10)
    parser.add_argument("--vote-threshold", type=int, default=25)
    parser.add_argument("--min-points", type=int, default=15)
    # parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--processes", "-J", type=int, default=1)
    parser.add_argument("--gather-threshold", nargs="+", type=float, default=[1, 1, 1])

    args = parser.parse_args()

    def per_catalog(work_dir, fakes, images, catalog, collection, detector, snr):
        os.makedirs(work_dir, exist_ok=True)
        search = File(f"{work_dir}/single_search")
        future, search = launch_for_outputs(
            single_search,
            velocity=args.velocity, 
            angle=args.angle,
            dx=args.dx,
            vote_threshold=args.vote_threshold,
            min_points=args.min_points,
            gather_threshold=args.gather_threshold,
            inputs=catalog + fakes + images,
            outputs=[search],
            work_dir=work_dir,
        )
        futures.append(future)

    def per_snr(work_dir, fakes, images, collection, detector, snr):
        os.makedirs(work_dir, exist_ok=True)
        catalog = File(os.path.join(work_dir, "catalog.pkl"))
        future, catalog = launch_for_outputs(
            detection,
            snr,
            inputs=images,
            outputs=[catalog],
            work_dir=work_dir,
        )
        futures.append(future)
        shuffled_catalog = File(os.path.join(work_dir, "catalog_shuffled.pkl"))
        future, shuffled_catalog = launch_for_outputs(
            shuffle,
            inputs=catalog,
            outputs=[shuffled_catalog],
            work_dir=work_dir,
        )
        futures.append(future)
        for c, subdir in zip([catalog, shuffled_catalog], ["unshuffled", "shuffled"]):
            per_catalog(os.path.join(work_dir, subdir), fakes, images, c, collection, detector, snr)

    def per_detector(work_dir, fakes, collection, detector):
        os.makedirs(work_dir, exist_ok=True)
        filtered_fakes = File(os.path.join(work_dir, "fakes.pkl"))
        future, filtered_fakes = launch_for_outputs(
            filter_fakes,
            collection, args.datasetType, detector,
            inputs=fakes,
            outputs=[filtered_fakes],
            work_dir=work_dir,
        )
        futures.append(future)
        images = File(os.path.join(work_dir, "images.pkl"))
        future, images = launch_for_outputs(
            get_images,
            collection, args.datasetType, detector,
            inputs=[],
            outputs=[images],
            work_dir=work_dir,
        )
        futures.append(future)
        for snr in args.snrs:
            per_snr(os.path.join(work_dir, f"snr_{snr}"), filtered_fakes, images, collection, detector, snr)

    def per_collection(work_dir, fakes, collection):
        os.makedirs(work_dir, exist_ok=True)
        for detector in args.detectors:
            per_detector(os.path.join(work_dir, f"detector_{detector}"), fakes, collection, detector)

    def launch_tasks(work_dir):
        os.makedirs(work_dir, exist_ok=True)
        future, fakes = launch_for_outputs(
            get_fakes,
            outputs=[File(os.path.join(work_dir, "fakes.pkl"))],
            work_dir=work_dir,
        )
        futures.append(future)
        for collection in args.collections:
            per_collection(os.path.join(work_dir, collection), fakes, collection)
    
    config, dfk = load_parsl(args.processes)
    
    futures = []
    try:
        launch_tasks(args.work_dir)
    except:
        dfk.cleanup()

    for future in futures:
        if future is not None:
            future.exception()

    dfk.cleanup()
    parsl.DataFlowKernelLoader.clear()


if __name__ == "__main__":
    main()
