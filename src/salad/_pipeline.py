import logging
import parsl
from parsl import Config, bash_app, python_app, join_app, File
from parsl.providers import LocalProvider
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
import os
import yaml
from functools import partial
from pathlib import Path
import re

logging.basicConfig()
log = logging.getLogger(__name__)

def run_command(
    command_line: str,
    inputs=(),
    outputs=(),
    stdout=None,
    stderr=None,
    parsl_resource_specification=None,
) -> str:
    return command_line

def command_generator(
        create_command, 
        inputs=(), 
        outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        skip_existing=True, 
        time_execution=True, 
        **kwargs
    ):
    fmt_dict = {}
    for i, input in enumerate(inputs):
        fmt_dict[f"input_{i}"] = input
    
    for i, output in enumerate(outputs):
        fmt_dict[f"output_{i}"] = output

    if skip_existing:
        skip_outputs = " && ".join(map(lambda x : f"test -f {x}", map(str, outputs)))
        if skip_outputs:
            skip_outputs = "{ \n" + skip_outputs + "\n } || "
    else:
        skip_outputs = ""
    
    fmt_dict.update(kwargs)
    print(fmt_dict)

    cmd = create_command(inputs=inputs, outputs=outputs, stdout=stdout, stderr=stderr)
    print(cmd)

    if time_execution:
        cmd = "/bin/time " + cmd

    script = "set -x\n" + skip_outputs + cmd.format(**fmt_dict)
    def f(inputs=inputs, outputs=outputs, stdout=stdout, stderr=stderr):
        return script
    setattr(f, "__name__", create_command.__name__)
    return f

def get_fakes(
        inputs=(),
        outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,
    ):
    cmd = f"salad fakes.get $REPO DEEP_fakes --fakesType asteroid --collections DEEP/fakes 1> {outputs[0]}"
    return cmd

def filter_fakes(
        collection, datasetType, detector, 
        inputs=(), outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs
    ):
    cmd = f"""salad fakes.filter {inputs[0]} {outputs[0]} --repo $REPO --datasetType {datasetType} --collections {collection} --where "instrument='DECam' and detector={detector}" """
    return cmd

def get_images(
        collection, datasetType, detector, 
        inputs=(), outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs        
):
    cmd = f"""salad images $REPO {datasetType} {outputs[0]} --collections {collection} --where "instrument='DECam' and detector={detector}" """
    return cmd

def detection(
        snr, 
        inputs=(), outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs                
):
    cmd = f"""salad detection {inputs[0]} {outputs[0]} --threshold {snr} --processes 1"""    
    return cmd

def project(
        v_min, v_max,
        angle_min, angle_max, 
        dx,
        inputs=(), outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs                        
):
    cmd = f"""salad project {inputs[0]} {outputs[0]} --velocity {v_min} {v_max} --angle {angle_min} {angle_max} --dx {dx}"""
    return cmd

def cluster_hough(
        dx,
        inputs=(), outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs
):
    cmd = f"""salad hough {inputs[0]} {outputs[0]} --dx {dx}"""
    return cmd

def find_clusters(
        threshold,
        inputs=(), outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs
):
    cmd = f"""salad find_clusters {inputs[0]} {outputs[0]} --threshold {threshold}"""
    return cmd

def refine(
        inputs=(), outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,  
):
    cmd = f"""salad refine {inputs[0]} {outputs[0]}"""
    return cmd

def gather(
        inputs=(), outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,        
):
    cmd = f"""salad gather {inputs[0]} {outputs[0]} --catalog {inputs[1]}"""
    return cmd

def recover(
        inputs=(), outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,                
):
    cmd = f"""salad fakes.recover {inputs[0]} {inputs[1]} {inputs[2]} {outputs[0]}"""
    return cmd

def detectable_in_catalog(
        inputs=(), outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,        
):
    cmd = f"""salad fakes.detectable_in_catalog {inputs[0]} {inputs[1]} {outputs[0]} --match-threshold 1 --match-threshold-unit arcsec"""
    return cmd
    
def detectable_in_search(
        inputs=(), outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,                
):
    cmd = f"""salad fakes.detectable_in_search {inputs[0]} {inputs[1]} {outputs[0]}"""
    return cmd

def join(inputs=(), outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,
):
    cmd = f"""salad cluster.join {inputs[0]} {outputs[0]} --catalog {inputs[1]} """
    return cmd

def split_clusters(
        split_format,
        inputs=(), 
        outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,
):
    cmd = f""" salad cluster.split {inputs[0]} --output-format {split_format} """
    return cmd

def get_cutouts(
        collection, datasetType, detector, 
        inputs=(), 
        outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,        
):
    cmd = f""" salad cutouts.get {inputs[0]} {outputs[0]} --repo $REPO --datasetType {datasetType} --collections {collection} --where "instrument='DECam' and detector={detector}" """
    return cmd

def forced_photometry(
        inputs=(), 
        outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,                
):
    cmd = f""" salad measure.forced {inputs[0]} {outputs[0]} """
    return cmd

def launch_for_outputs(function, *args, **kwargs):
    outputs = kwargs.get("outputs", [])
    outputs_not_exist = list(map(lambda output : not os.path.exists(output.path), outputs))
    if any(outputs_not_exist):
        for output in outputs:
            os.makedirs(os.path.dirname(output.path), exist_ok=True)
        future = bash_app(function)(*args, **kwargs)
        return future, future.outputs
    else:
        log.info("skipping %s since outputs=%s already exist", function, outputs)
        return None, outputs

def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser(prog=__name__)
    # parser.add_argument("config", type=str)
    parser.add_argument("pipeline", type=str)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--processes", "-J", type=int, default=1)

    args = parser.parse_args()
    # with open(args.config, "r") as fd:
    #     pipeline_config = yaml.load(fd, Loader=yaml.SafeLoader)

    os.environ['OMP_NUM_THREADS'] = "1"
    os.environ['MKL_NUM_THREADS'] = "1"
    os.environ['NUMBA_NUM_THREADS'] = "1"
    os.environ['NUMEXPR_NUM_THREADS'] = "1"

    config = Config(
        executors=[
            HighThroughputExecutor(
                label="htex",
                provider=LocalProvider(
                    min_blocks=0,
                    max_blocks=args.processes,
                    worker_init=f"""
source {os.environ['SALAD_DIR']}/bin/setup.sh
load_lsst
export PATH={os.environ['SALAD_DIR']}/bin:$PATH
export PATH={os.environ['SALAD_DIR']}/env/bin:$PATH
which process_worker_pool.py
"""
                ),
                max_workers=1,
                encrypted=True,
            )
        ]
    )
    dfk = parsl.load(config)

    # func = command_generator(get_fakes, outputs=[File("fakes.pkl")])
    # bash_app(func)(outputs=[File("fakes.pkl")], )

    collections = ["DEEP/20190403/A0c"]#, "DEEP/20210504/A0f"]
    detectors = [1] # [i for i in range(1, 63)]
    snrs = [2] # [2.5]# [5, 4.5, 4, 3.5, 3]
    datasetType = "differenceExp"
    velocity = [0.1, 0.5]
    angle = [180 - 60, 180 + 60]
    dx = 5 #10
    vote_threshold = 25

    work_dir = "tmp"
    futures = []
    
    if args.pipeline == "search":
        future, fakes = launch_for_outputs(
            get_fakes,
            outputs=[File(f"{work_dir}/fakes.pkl")],
        )
        futures.append(future)
        for collection in collections:
            for detector in detectors:
                c = f"{work_dir}/{collection}/detector_{detector}"
                filtered_fakes = File(f"{c}/fakes.pkl")
                future, filtered_fakes = launch_for_outputs(
                    filter_fakes,
                    collection, datasetType, detector,
                    inputs=fakes,
                    outputs=[filtered_fakes]
                )
                futures.append(future)
                images = File(f"{c}/images.pkl")
                future, images = launch_for_outputs(
                    get_images,
                    collection, datasetType, detector,
                    inputs=[],
                    outputs=[images]
                )
                futures.append(future)
                for snr in snrs:
                    d = f"{c}/snr_{snr}"
                    catalog = File(f"{d}/catalog.pkl")
                    future, catalog = launch_for_outputs(
                        detection,
                        snr,
                        inputs=images,
                        outputs=[catalog]
                    )
                    futures.append(future)
                    projection = File(f"{d}/projection.pkl")
                    future, projection = launch_for_outputs(
                        project,
                        velocity[0], velocity[1], angle[0], angle[1], dx,
                        inputs=catalog,
                        outputs=[projection]
                    )
                    futures.append(future)
                    hough = File(f"{d}/hough.pkl")
                    future, hough = launch_for_outputs(
                        cluster_hough,
                        dx,
                        inputs=projection,
                        outputs=[hough]
                    )
                    futures.append(future)
                    clusters = File(f"{d}/clusters.pkl")
                    future, clusters = launch_for_outputs(
                        find_clusters,
                        vote_threshold,
                        inputs=hough, outputs=[clusters]
                    )
                    futures.append(future)
                    lines_1 = File(f"{d}/lines_1.pkl")
                    future, lines_1 = launch_for_outputs(
                        refine,
                        inputs=clusters,
                        outputs=[lines_1]
                    )
                    futures.append(future)
                    gather_1 = File(f"{d}/gather_1.pkl")
                    future, gather_1 = launch_for_outputs(
                        gather,
                        inputs=lines_1 + catalog,
                        outputs=[gather_1]
                    )
                    futures.append(future)
                    lines_2 = File(f"{d}/lines_2.pkl")
                    future, lines_2 = launch_for_outputs(
                        refine,
                        inputs=gather_1,
                        outputs=[lines_2]
                    )
                    futures.append(future)
                    gather_2 = File(f"{d}/gather_2.pkl")
                    future, gather_2 = launch_for_outputs(
                        gather,
                        inputs=lines_2 + catalog,
                        outputs=[gather_2]
                    )
                    futures.append(future)
                    recovery = File(f"{d}/recovery.pkl")
                    future, recovery = launch_for_outputs(
                        recover,
                        inputs=filtered_fakes + gather_2 + catalog,
                        outputs=[recovery]
                    )
                    futures.append(future)
                    catalog_detectable = File(f"{d}/catalog_detectable.pkl")
                    future, catalog_detectable = launch_for_outputs(
                        detectable_in_catalog,
                        inputs=filtered_fakes + catalog,
                        outputs=[catalog_detectable]
                    )
                    futures.append(future)
                    search_detectable = File(f"{d}/search_detectable.pkl")
                    future, search_detectable = launch_for_outputs(
                        detectable_in_search,
                        inputs=filtered_fakes + hough,
                        outputs=[search_detectable]
                    )
                    futures.append(future)
                    # join
                    joined = File(f"{d}/gather_2_joined.pkl")
                    future, joined = launch_for_outputs(
                        join,
                        inputs=gather_2 + catalog,
                        outputs=[joined]
                    )
                    futures.append(future)
                    # split
                    clusters_dir = f"{d}/gather_2_clusters"
                    split_format = f"{clusters_dir}/cluster_%i.pkl"
                    future = bash_app(split_clusters)(split_format, inputs=joined)
                    futures.append(future)

    elif args.pipeline == "measure":
        for collection in collections:
            for detector in detectors:
                c = f"{work_dir}/{collection}/detector_{detector}"
                for snr in snrs:
                    d = f"{c}/snr_{snr}"
                    clusters_dir = f"{d}/gather_2_clusters"
                    for cluster_path in Path(clusters_dir).glob("*.pkl"):
                        if re.compile(r"cluster_\d+.pkl").match(cluster_path.name):
                            cluster = File(str(cluster_path))
                            cutouts_path = cluster_path.parent / cluster_path.name.replace(".pkl", "_cutouts.pkl")
                            cutouts = File(str(cutouts_path))
                            future, cutouts = launch_for_outputs(
                                get_cutouts,
                                collection, datasetType, detector,
                                inputs=[cluster],
                                outputs=[cutouts]
                            )
                            futures.append(future)

                            forced_path = cluster_path.parent / cluster_path.name.replace(".pkl", "_forced.pkl")
                            forced = File(str(forced_path))
                            future, forced = launch_for_outputs(
                                forced_photometry,
                                inputs=cutouts,
                                outputs=[forced]
                            )
                            futures.append(future)


                    # getting cutouts is slow
                    # but it's too complicated to parallelize here

                    # # cutouts
                    # cutouts = File(f"{clusters_dir}/cluster_%i_with_cutouts.pkl")
                    # future, cutouts = launch_for_outputs(
                    #     get_cutouts, # this will be a join app
                    #     collection, datasetType, detector,
                    #     inputs=split,

                    # )
                    # # measure
                    # forced_measurements = File(f"{d}/gather_2_clusters/cluster_%i_forced.pkl")
                    # future, forced_measurements = launch_for_outputs(

                    # )


    for future in futures:
        if future is not None:
            future.exception()

    dfk.cleanup()
    parsl.DataFlowKernelLoader.clear()

if __name__ == "__main__":
    main()
