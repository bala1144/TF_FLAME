#!/bin/sh

# default parameters
BID=$1
NODE_CONFIG=./condor/default_node_config.sub

# # # for training
CONFIG=sentence21_ns12.pkl
NODE_SCRIPT=./condor/job.sh
# set parameters
if [ -n "$1" ]; then BID=${1}; fi
if [ -n "$2" ]; then CONFIG=${2}; fi
if [ -n "$3" ]; then NODE_CONFIG=${3}; fi
if [ -n "$4" ]; then NODE_SCRIPT=${4}; fi

# start node and execute script
echo 'Executing:' ${NODE_SCRIPT} ${CONFIG}
echo 'BID:' ${BID}
condor_submit_bid ${BID} ${NODE_CONFIG} -append "arguments = ${NODE_SCRIPT} ${CONFIG}"