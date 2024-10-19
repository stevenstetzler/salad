import parsl
from parsl import Config
import os
from parsl.providers import LocalProvider
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor

def load_parsl(processes):
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
    dfk = parsl.load(config)
    return config, dfk

