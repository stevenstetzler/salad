import parsl
from parsl import Config
import os
from parsl.providers import LocalProvider, SlurmProvider
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor

def _load_klone(processes):
    config = Config(
        executors=[
            HighThroughputExecutor(
                label="htex",
                provider=SlurmProvider(
                    partition=os.environ.get("SALAD_PARTITION", "compute-bigmem"),
                    account=os.environ.get("SALAD_ACCOUNT", "astro"),
                    nodes_per_block=1,
                    cores_per_node=int(os.environ.get("SALAD_CORES", "1")),
                    mem_per_node=int(os.environ.get("SALAD_MEMORY", "8")),
                    min_blocks=0,
                    max_blocks=processes,
                    walltime="04:00:00",
                    exclusive=False,
                    worker_init=f"""
source {os.environ['SALAD_DIR']}/bin/setup.sh
load_lsst
export PATH={os.environ['SALAD_DIR']}/bin:$PATH
export PATH={os.environ['SALAD_DIR']}/env/bin:$PATH
which process_worker_pool.py
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
"""
                ),
                encrypted=True,
            )
        ]
    )
    return config    


def _load_local(processes):
    os.environ['OMP_NUM_THREADS'] = "1"
    os.environ['MKL_NUM_THREADS'] = "1"
    os.environ['NUMBA_NUM_THREADS'] = "1"
    os.environ['NUMEXPR_MAX_THREADS'] = "1"

    config = Config(
        executors=[
            HighThroughputExecutor(
                label="htex",
                provider=LocalProvider(
                    min_blocks=0,
                    max_blocks=processes,
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
    return config    

def load_parsl(processes):
    site = os.environ.get("SALAD_SITE", "local")
    if site == "local":
        config = _load_local(processes)
    elif site == "klone":
        config = _load_klone(processes)
    else:
        raise Exception(f"site '{site}' not supported")
    
    dfk = parsl.load(config)
    return config, dfk

