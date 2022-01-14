#!/bin/sh

# # default parameters
BID=$1
POSTFIX=${2:-def}
CONFIG_PATH=./configs/vqvae/subsplit_v42_mlp_size_vqvae_PEcodebooks_train_bsexp
NODE_CONFIG=./condor/default_node_config.sub
NODE_SCRIPT=./condor/job.sh

# set parameters
if [ -n "$1" ]; then BID=${1}; fi
if [ -n "$2" ]; then CONFIG=${2}; fi
if [ -n "$3" ]; then NODE_CONFIG=${3}; fi
if [ -n "$4" ]; then NODE_SCRIPT=${4}; fi

# start node and execute script
for CONFIG in ${CONFIG_PATH}/*.yaml; do
    echo 'Executing:' ${NODE_SCRIPT} ${CONFIG}
    echo 'BID:' ${BID}
    condor_submit_bid ${BID} ${NODE_CONFIG} -append "arguments = ${NODE_SCRIPT} ${CONFIG} ${POSTFIX}"
done