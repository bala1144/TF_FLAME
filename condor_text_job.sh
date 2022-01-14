#!/bin/sh

# # default parameters
BID=$1
NODE_CONFIG=./condor/default_node_config.sub
MULTI_CONFIG_FILE=./all_voca_data_splits_rerun2.txt

# FOR THE RESUME JOB
NODE_SCRIPT=./condor/job.sh

# start node and execute script
while read CONFIG; do
  echo 'Executing:' ${NODE_SCRIPT} ${CONFIG}
  echo 'BID:' ${BID}
  condor_submit_bid ${BID} ${NODE_CONFIG} -append "arguments = ${NODE_SCRIPT} ${CONFIG}"
done <${MULTI_CONFIG_FILE}