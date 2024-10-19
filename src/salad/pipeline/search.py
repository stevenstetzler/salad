import logging
import parsl
from parsl import bash_app, File
import os
from .parsl_config import load_parsl
from .tasks import (
    launch_for_outputs, 
    detection, project, 
    cluster_hough, find_clusters, 
    refine, gather, filter_clusters, join, split_clusters,
    recover, detectable_in_catalog, detectable_in_search,
    filter_fakes, get_images, get_fakes, shuffle, summary, deduplicate
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
    parser.add_argument("--datasetType", default="differenceExp")
    parser.add_argument("--velocity", nargs=2, type=float, default=[0.1, 0.5])
    parser.add_argument("--angle", nargs=1, type=float, default=[120, 240])
    parser.add_argument("--dx", type=float, default=10)
    parser.add_argument("--vote-threshold", type=int, default=25)
    parser.add_argument("--filter-velocity", nargs=2, type=float, default=None)
    parser.add_argument("--filter-angle", nargs=2, type=float, default=None)
    parser.add_argument("--min-points", type=int, default=15)
    # parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--processes", "-J", type=int, default=1)
    # parser.add_argument("--gather-threshold", nargs="+", type=float, default=[1, 1, 1])
    parser.add_argument("--min-gather-threshold", type=float, default=1)
    parser.add_argument("--cutout-width", type=int, default=50)
    parser.add_argument("--cutout-height", type=int, default=50)
    parser.add_argument("--origin-threshold", type=float, default=5)
    parser.add_argument("--beta-threshold", type=float, default=1)

    args = parser.parse_args()

    if args.filter_velocity is None:
        args.filter_velocity = args.velocity
    
    if args.filter_angle is None:
        args.filter_angle = args.angle        

    def per_catalog(work_dir, fakes, images, catalog, collection, detector, snr):
        os.makedirs(work_dir, exist_ok=True)
        projection = File(f"{work_dir}/projection.pkl")
        future, projection = launch_for_outputs(
            project,
            args.velocity[0], args.velocity[1], args.angle[0], args.angle[1], args.dx,
            inputs=catalog,
            outputs=[projection],
            work_dir=work_dir,
        )
        futures.append(future)
        hough = File(f"{work_dir}/hough.pkl")
        future, hough = launch_for_outputs(
            cluster_hough,
            args.dx,
            inputs=projection,
            outputs=[hough],
            work_dir=work_dir,
        )
        futures.append(future)

        summaries_for = []
        to_recover = []

        clusters = File(f"{work_dir}/clusters.pkl")
        future, clusters = launch_for_outputs(
            find_clusters,
            args.vote_threshold,
            inputs=hough, 
            outputs=[clusters],
            work_dir=work_dir,
        )
        futures.append(future)
        summaries_for.extend(clusters)
        to_recover.extend(clusters)

        lines = File(f"{work_dir}/refined.pkl")
        future, lines = launch_for_outputs(
            refine,
            inputs=clusters,
            outputs=[lines],
            work_dir=work_dir,
        )
        step = 0
        refine_output = lines
        gather_threshold = args.dx
        while gather_threshold > args.min_gather_threshold:
            gather_threshold /= 2
            gather_threshold = max(gather_threshold, args.min_gather_threshold)
            # gather + refine
            gather_output = File(f"{work_dir}/gathered_{step}.pkl")
            future, gather_output = launch_for_outputs(
                gather,
                threshold=gather_threshold,
                inputs=refine_output + catalog,
                outputs=[gather_output],
                work_dir=work_dir,
            )
            futures.append(future)
            to_recover.extend(gather_output)
            summaries_for.extend(gather_output)
            refine_output = File(f"{work_dir}/refined_{step}.pkl")
            future, refine_output = launch_for_outputs(
                refine,
                inputs=gather_output,
                outputs=[refine_output],
                work_dir=work_dir,
            )
            futures.append(future)  
            step += 1          

        gather_output = File(f"{work_dir}/gathered_{step}.pkl")
        future, gather_output = launch_for_outputs(
            gather,
            threshold=gather_threshold,
            inputs=refine_output + catalog,
            outputs=[gather_output],
            work_dir=work_dir,
        )
        futures.append(future)  
        summaries_for.extend(gather_output)
        to_recover.extend(gather_output)

        refine_output = File(f"{work_dir}/refined_{step}.pkl")
        future, refine_output = launch_for_outputs(
            refine,
            inputs=gather_output,
            outputs=[refine_output],
            work_dir=work_dir,
        )
        futures.append(future) 

        dedup = File(f"{work_dir}/deduplicated.pkl")
        future, dedup = launch_for_outputs(
            deduplicate,
            args.origin_threshold, args.beta_threshold,
            inputs=gather_output + images,
            outputs=[dedup],
            work_dir=work_dir
        )
        futures.append(future)
        to_recover.extend(dedup)

        filtered_clusters = File(f"{work_dir}/filtered.pkl")
        future, filtered_clusters = launch_for_outputs(
            filter_clusters,
            args.filter_velocity, args.filter_angle, args.min_points,
            inputs=dedup,
            outputs=[filtered_clusters],
            work_dir=work_dir,
        )
        futures.append(future)
        to_recover.extend(filtered_clusters)
        summaries_for.extend(filtered_clusters)

        summaries_for = filtered_clusters

        recoveries = []
        for f in to_recover:
            p = os.path.join(
                os.path.dirname(f.filepath), os.path.basename(f.filepath).replace(".pkl", "_recovery.pkl")
            )
            future, recovery = launch_for_outputs(
                recover,
                inputs=fakes + [f] + catalog + hough,
                outputs=[File(p)],
                work_dir=work_dir,
            )
            futures.append(future)
            recoveries.extend(recovery)

        return summaries_for
        # lines_1 = File(f"{work_dir}/refine_results_1.pkl")
        # future, lines_1 = launch_for_outputs(
        #     refine,
        #     inputs=clusters,
        #     outputs=[lines_1],
        #     work_dir=work_dir,
        # )
        # futures.append(future)
        # gather_1 = File(f"{work_dir}/refined_clusters_1.pkl")
        # future, gather_1 = launch_for_outputs(
        #     gather,
        #     threshold=args.gather_threshold[0],
        #     inputs=lines_1 + catalog,
        #     outputs=[gather_1],
        #     work_dir=work_dir,
        # )
        # futures.append(future)
        # lines_2 = File(f"{work_dir}/refine_results_2.pkl")
        # future, lines_2 = launch_for_outputs(
        #     refine,
        #     inputs=gather_1,
        #     outputs=[lines_2],
        #     work_dir=work_dir,
        # )
        # futures.append(future)
        # gather_2 = File(f"{work_dir}/refined_clusters_2.pkl")
        # future, gather_2 = launch_for_outputs(
        #     gather,
        #     threshold=args.gather_threshold[1],
        #     inputs=lines_2 + catalog,
        #     outputs=[gather_2],
        #     work_dir=work_dir,
        # )
        # lines_3 = File(f"{work_dir}/refine_results_3.pkl")
        # future, lines_3 = launch_for_outputs(
        #     refine,
        #     inputs=gather_2,
        #     outputs=[lines_3],
        #     work_dir=work_dir,
        # )
        # futures.append(future)
        # gather_3 = File(f"{work_dir}/refined_clusters_3.pkl")
        # future, gather_3 = launch_for_outputs(
        #     gather,
        #     threshold=args.gather_threshold[2],
        #     inputs=lines_3 + catalog,
        #     outputs=[gather_3],
        #     work_dir=work_dir,
        # )
        # futures.append(future)
        # recovery = File(f"{work_dir}/recovery.pkl")
        # future, recovery = launch_for_outputs(
        #     recover,
        #     inputs=fakes + gather_2 + catalog,
        #     outputs=[recovery],
        #     work_dir=work_dir,
        # )
        # futures.append(future)
        # catalog_detectable = File(f"{work_dir}/catalog_detectable.pkl")
        # future, catalog_detectable = launch_for_outputs(
        #     detectable_in_catalog,
        #     inputs=fakes + catalog,
        #     outputs=[catalog_detectable],
        #     work_dir=work_dir,
        # )
        # futures.append(future)
        # search_detectable = File(f"{work_dir}/search_detectable.pkl")
        # future, search_detectable = launch_for_outputs(
        #     detectable_in_search,
        #     inputs=fakes + hough,
        #     outputs=[search_detectable],
        #     work_dir=work_dir,
        # )
        # futures.append(future)
        # # filter
        # filtered_clusters = File(f"{work_dir}/refined_clusters_2_filtered.pkl")
        # future, filtered_clusters = launch_for_outputs(
        #     filter_clusters,
        #     args.velocity, args.angle, args.min_points,
        #     inputs=gather_2,
        #     outputs=[filtered_clusters],
        #     work_dir=work_dir,
        # )
        # futures.append(future)
        # # recover filter
        # recovery_filtered = File(f"{work_dir}/recovery_filtered.pkl")
        # future, recovery = launch_for_outputs(
        #     recover,
        #     inputs=fakes + filtered_clusters + catalog,
        #     outputs=[recovery_filtered],
        #     work_dir=work_dir,
        # )
        # futures.append(future)
        # # join
        # joined = File(f"{work_dir}/refined_clusters_2_filtered_joined.pkl")
        # future, joined = launch_for_outputs(
        #     join,
        #     inputs=filtered_clusters + catalog,
        #     outputs=[joined],
        #     work_dir=work_dir,
        # )
        # futures.append(future)
        # # split
        # clusters_dir = f"{work_dir}/refined_clusters_2_filtered_joined_split"
        # split_format = f"{clusters_dir}/%i/cluster.pkl"
        # future = bash_app(split_clusters)(
        #     split_format, 
        #     inputs=joined,
        #     stdout=os.path.join(work_dir, "split_clusters.stdout"),
        #     stderr=os.path.join(work_dir, "split_clusters.stderr"),
        # )
        # futures.append(future)

    def per_snr(work_dir, fakes, images, collection, detector, snr):
        os.makedirs(work_dir, exist_ok=True)
        for subdir in ["regular", "shuffled"]:
            os.makedirs(os.path.join(work_dir, subdir), exist_ok=True)
        catalog = File(os.path.join(work_dir, "regular", "catalog.pkl"))
        future, catalog = launch_for_outputs(
            detection,
            snr,
            inputs=images,
            outputs=[catalog],
            work_dir=work_dir,
        )
        futures.append(future)
        catalog_no_masks = File(os.path.join(work_dir, "regular", "catalog_no_masks.pkl"))
        future, catalog_no_masks = launch_for_outputs(
            detection,
            snr,
            inputs=images,
            no_masks=True,
            outputs=[catalog_no_masks],
            work_dir=work_dir,
        )
        futures.append(future)

        shuffled_catalog = File(os.path.join(work_dir, "shuffled", "catalog.pkl"))
        future, shuffled_catalog = launch_for_outputs(
            shuffle,
            inputs=catalog,
            outputs=[shuffled_catalog],
            work_dir=work_dir,
        )
        futures.append(future)
        summaries_for = []
        for c, subdir in zip([catalog, shuffled_catalog], ["regular", "shuffled"]):
            summaries_for.extend(per_catalog(os.path.join(work_dir, subdir, args.label), fakes, images, c, collection, detector, snr))
        
        return summaries_for

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
        summaries_for = []
        for snr in args.snrs:
            summaries_for.extend(per_snr(os.path.join(work_dir, f"snr_{snr}"), filtered_fakes, images, collection, detector, snr))

        # do summaries all at once to reduce IO on the disks
        summaries = []
        # I need to pop tasks that have failed here somehow...
        for f in summaries_for:
            p = os.path.join(
                os.path.dirname(f.filepath), os.path.basename(f.filepath).replace(".pkl", "_summary.pkl")
            )
            summaries.append(File(p))
        
        future, summaries = launch_for_outputs(
            summary,
            args.cutout_width, args.cutout_height,
            inputs=images + summaries_for,
            outputs=summaries,
            work_dir=work_dir,
        )
        futures.append(future)

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
