#!/usr/bin/env bash
# set -euo pipefail

BIN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

source /epyc/ssd/users/stevengs/opt_lsst/bin/opt_lsst.sh

pathadd() {
    if [ -d "$1" ] && [[ ":$PATH:" != *":$1:"* ]]; then
        PATH="$1:${PATH:+"$PATH:"}"
    fi
}
pythonpathadd() {
    if [ -d "$1" ] && [[ ":$PYTHONPATH:" != *":$1:"* ]]; then
        PYTHONPATH="$1:${PYTHONPATH:+"$PYTHONPATH:"}"
    fi
}
export SALAD_DIR=$(dirname $BIN_DIR)

source $SALAD_DIR/config.sh
export REPO="${SALAD_DIR}/data"

J=1
export OMP_NUM_THREADS=$J
export MKL_NUM_THREADS=$J 
export NUMBA_NUM_THREADS=$J
export NUMEXPR_NUM_THREADS=$J
renice -n 10 $$ # lower the priority of any commands started from this shell

pathadd "$BIN_DIR"

function load_lsst() {
    opt_lsst setup w_2024_09
    pathadd "${SALAD_DIR}/env/bin/salad"
    pathadd "${SALAD_DIR}/env/bin/process_worker_pool.py"
    pythonpathadd "${SALAD_DIR}/src"
}

function load_env() {
    source $SALAD_DIR/env/bin/activate
    pathadd "${SALAD_DIR}/env/bin"
    export PYTHONPATH="${SALAD_DIR}/src"
}

