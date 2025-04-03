import parsl
from parsl import bash_app
import os
import logging

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
        cmd = "/bin/time --verbose " + cmd

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
    cmd = f" /bin/time --verbose salad fakes.get $REPO DEEP_fakes --collections DEEP/fakes 1> {outputs[0]}"
    return cmd

def filter_fakes(
        collection, datasetType, detector, 
        inputs=(), outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs
    ):
    cmd = f""" /bin/time --verbose salad fakes.filter {inputs[0]} {outputs[0]} --repo $REPO --datasetType {datasetType} --collections {collection} --where "instrument='DECam' and detector={detector}" """
    return cmd

def get_images(
        collection, datasetType, detector, 
        inputs=(), outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs        
):
    cmd = f""" /bin/time --verbose salad images $REPO {datasetType} {outputs[0]} --collections {collection} --where "instrument='DECam' and detector={detector}" """
    return cmd

def detection(
        snr, 
        no_masks=False,
        inputs=(), outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs                
):
    cmd = f""" /bin/time --verbose salad detection {inputs[0]} {outputs[0]} --threshold {snr} --processes 1 """
    if no_masks:
        cmd += " --no-masks "
    return cmd

def shuffle(
        inputs=(), outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs
):
    cmd = f""" /bin/time --verbose salad shuffle {inputs[0]} {outputs[0]}"""
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
    cmd = f""" /bin/time --verbose salad project {inputs[0]} {outputs[0]} --velocity {v_min} {v_max} --angle {angle_min} {angle_max} --dx {dx}"""
    return cmd

def cluster_hough(
        dx,
        inputs=(), outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs
):
    cmd = f""" /bin/time --verbose salad hough {inputs[0]} {outputs[0]} --dx {dx}"""
    return cmd

def find_clusters(
        threshold,
        threshold_type='votes',
        inputs=(), outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs
):
    cmd = f""" /bin/time --verbose salad find_clusters {inputs[0]} {outputs[0]} --threshold {threshold} --threshold-type {threshold_type} """
    return cmd

def refine(
        inputs=(), outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,  
):
    cmd = f""" /bin/time --verbose salad refine {inputs[0]} {outputs[0]}"""
    return cmd

def gather(
        threshold,
        inputs=(), outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,        
):
    cmd = f""" /bin/time --verbose salad gather {inputs[0]} {outputs[0]} --catalog {inputs[1]} --threshold {threshold}"""
    return cmd

def recover(
        inputs=(), outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,                
):
    cmd = f""" /bin/time --verbose salad fakes.recover --fakes {inputs[0]} --clusters {inputs[1]} --catalog {inputs[2]} --hough {inputs[3]} {outputs[0]}"""
    return cmd

def match_fakes_catalog(
        fakes,
        catalog,
        inputs=(), outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,                
):
    cmd = f""" /bin/time --verbose salad fakes.recover {outputs[0]} --action match_catalog --fakes {fakes} --catalog {catalog} """
    return cmd

def match_fakes_clusters_line(
        fakes,
        clusters,
        inputs=(), outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,                
):
    cmd = f""" /bin/time --verbose salad fakes.recover {outputs[0]} --action match_clusters_line --fakes {fakes} --clusters {clusters} """
    return cmd

def match_fakes_clusters_points(
        fakes,
        clusters,
        inputs=(), outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,                
):
    cmd = f""" /bin/time --verbose salad fakes.recover {outputs[0]} --action match_clusters_points --fakes {fakes} --clusters {clusters} """
    return cmd

def fakes_info(
        fakes,
        inputs=(), outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,                
):
    cmd = f""" /bin/time --verbose salad fakes.recover {outputs[0]} --action fakes_info --fakes {fakes} """
    return cmd

def detectable_in_catalog(
        inputs=(), outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,        
):
    cmd = f""" /bin/time --verbose salad fakes.detectable_in_catalog {inputs[0]} {inputs[1]} {outputs[0]} --match-threshold 1 --match-threshold-unit arcsec"""
    return cmd
    
def detectable_in_search(
        inputs=(), outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,                
):
    cmd = f""" /bin/time --verbose salad fakes.detectable_in_search {inputs[0]} {inputs[1]} {outputs[0]}"""
    return cmd

def join(inputs=(), outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,
):
    cmd = f""" /bin/time --verbose salad cluster.join {inputs[0]} {outputs[0]} --catalog {inputs[1]} """
    return cmd

def split_clusters(
        split_format,
        inputs=(), 
        outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,
):
    cmd = f"""  /bin/time --verbose salad cluster.split {inputs[0]} --output-format {split_format} """
    return cmd

def get_cutouts(
        collection, datasetType, detector, 
        inputs=(), 
        outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,        
):
    cmd = f"""  /bin/time --verbose salad cutouts.get {inputs[0]} {outputs[0]} --repo $REPO --datasetType {datasetType} --collections {collection} --where "instrument='DECam' and detector={detector}" """
    return cmd

def forced_points(
        inputs=(), 
        outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,                
):
    cmd = f"""  /bin/time --verbose salad measure.forced_points {inputs[0]} {outputs[0]} """
    return cmd

def forced_line(
        images,
        inputs=(), 
        outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,                
):
    cmd = f"""  /bin/time --verbose salad measure.forced_line {inputs[0]} {outputs[0]} --images {images}"""
    return cmd

def coadd_points(
        inputs=(), 
        outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,                
):
    cmd = f"""  /bin/time --verbose salad measure.coadd_points {inputs[0]} {outputs[0]} """
    return cmd

def coadd_line(
        images,
        inputs=(), 
        outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,                
):
    cmd = f"""  /bin/time --verbose salad measure.coadd_line {inputs[0]} {outputs[0]} --images {images}"""
    return cmd

def fit_line(
        images,
        inputs=(), 
        outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,                
):
    cmd = f"""  /bin/time --verbose salad measure.fitting_line {inputs[0]} {outputs[0]} --images {images}"""
    return cmd


def filter_clusters(
        velocity, angle, min_points,
        inputs=(), 
        outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,                
):
    cmd = f"""  /bin/time --verbose salad cluster.filter {inputs[0]} {outputs[0]} --velocity {velocity[0]} {velocity[1]} --angle {angle[0]} {angle[1]} --min-points {min_points} """
    return cmd


def plot_cutouts(
        output_format=None,
        inputs=(), 
        outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,                
):
    if output_format:
        output_format = f"--output-format {output_format}"
    else:
        output_format = ""
    cmd = f"""  /bin/time --verbose salad cutouts.plot {inputs[0]} {outputs[0]} {output_format} """
    return cmd

def single_search(
        velocity, angle, 
        dx, vote_threshold, min_points, gather_threshold,
        inputs=(), 
        outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,                
):
    cmd = f"""  /bin/time --verbose salad single_search {inputs[0]} {outputs[0]} --fakes {inputs[1]} --images {inputs[2]} --velocity {velocity[0]} {velocity[1]} --angle {angle[0]} {angle[1]} --min-points {min_points} --dx {dx} --vote-threshold {vote_threshold} --min-points {min_points} --gather-threshold {gather_threshold[0]} {gather_threshold[1]}"""
    return cmd

def summary(
        width, height,
        inputs=(), 
        outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,                
):
    cmd = f"""  /bin/time --verbose salad analysis.summary {' '.join(map(str, inputs[1:]))} --images {inputs[0]} --width {width} --height {height}"""
    return cmd

def deduplicate(
        origin_threshold, beta_threshold,
        inputs=(), 
        outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,                
):
    cmd = f"""  /bin/time --verbose salad deduplicate {inputs[0]} {outputs[0]} --images {inputs[1]} --origin-threshold {origin_threshold} --beta-threshold {beta_threshold}"""
    return cmd


def plot(
        plot_type,
        inputs=(), 
        outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,                
):
    cmd = f"""  /bin/time --verbose salad analysis.plot {inputs[0]} {outputs[0]} --plot-type {plot_type} """
    return cmd



def search(
        images,
        catalog,
        dx,
        velocity,
        angle,
        threshold,
        threshold_type,
        inputs=(), 
        outputs=(), 
        stdout=parsl.AUTO_LOGNAME, 
        stderr=parsl.AUTO_LOGNAME, 
        **kwargs,                
):
    cmd = f"""  /bin/time --verbose salad search --images {images} --catalog {catalog} --dx {dx} --velocity {velocity[0]} {velocity[1]} --angle {angle[0]} {angle[1]} --threshold {threshold} --threshold-type {threshold_type} --output-clusters {outputs[0]} --output-gathered {outputs[1]} """
    return cmd


def launch_for_outputs(function, *args, **kwargs):
    outputs = kwargs.get("outputs", [])
    work_dir = kwargs.get("work_dir", None)
    if work_dir:
        stdout = kwargs.pop("stdout", os.path.join(work_dir, f"{function.__name__}.stdout"))
        stderr = kwargs.pop("stderr", os.path.join(work_dir, f"{function.__name__}.stderr"))
    else:
        stdout = kwargs.pop("stdout", parsl.AUTO_LOGNAME)
        stderr = kwargs.pop("stderr", parsl.AUTO_LOGNAME)

    outputs_not_exist = list(map(lambda output : not os.path.exists(output.path), outputs))
    if any(outputs_not_exist):
        for output in outputs:
            os.makedirs(os.path.dirname(output.path), exist_ok=True)
        future = bash_app(function)(*args, stdout=stdout, stderr=stderr, **kwargs)
        return future, future.outputs
    else:
        log.info("skipping %s since outputs=%s already exist", function, outputs)
        return None, outputs
    
