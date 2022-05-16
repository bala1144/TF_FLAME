#!/bin/sh

CONFIG=${1}
POSTFIX=${2:-def}

#PYTHON_ENV=/home/wzielonka/miniconda3/etc/profile.d/conda.sh
PYTHON_ENV=/home/bthambiraja/miniconda3/etc/profile.d/conda.sh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:$PATH
export HOME="/home/bthambiraja"

echo 'START data processing for'

source /etc/profile.d/modules.sh

echo 'ACTIVATE CONDA flame_fit_2'
source ${PYTHON_ENV}
conda activate flame_fit_2

nvidia-smi

echo 'RUN SCRIPT'
echo 'ScriptDir' ${SCRIPT_DIR}
echo 'CONFIG: ' ${CONFIG}
cd ${SCRIPT_DIR}/..
echo "Script executed from: ${PWD}"
python ./fitting_from_dataset_persubseqwise_noshape_optim.py -d ${CONFIG}