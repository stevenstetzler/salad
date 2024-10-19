#!/usr/bin/env bash
set -ex

BIN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
_log=$(dirname ${BIN_DIR})/logs/notebooks/$(echo $@ | awk '{print $2}').log
mkdir -p $(dirname ${_log})
echo "${_log}"
{
    which python
    source ${BIN_DIR}/setup.sh
    which python
    set +x
    load_lsst
    set -x
    which python
    exec python -m ipykernel $@
} 1> ${_log} 2>&1