#!/bin/bash

# Get array of cluster names
set -e

source clusters_names 

print_help (){
printf "$0 | [server_name]: \n\nLaunch the computation from experiment experiment_folder to server server_name and have results in ../results \n\n"
}

SERVER_NAME=$1

case "${SERVER_NAME}" in
  lyon) clusters=${clusters_lyon[@]};;
  grenoble) clusters=${clusters_grenoble[@]} ;;
  rennes) clusters=${clusters_rennes[@]} ;;
  nancy) clusters=${clusters_nancy[@]} ;;
  sophia) clusters=${clusters_sophia[@]} ;;
  test_server) clusters=("test");;
  *) print_help ; exit 0 ;;
esac

cd ~/ReproducibleRL/code/g5k_tools/

GREEN='\033[1;32m'
NC='\033[0m'

set +e # allow fail, e.g. for exotic jobs for instance

if [ ! -d ~/miniconda3 ]; then
    cd 
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash ~/Miniconda3-latest-Linux-x86_64.sh -bc 
    source ~/.bashrc
    conda tos accept
fi

# Loop over each cluster
for cluster in $(echo "${clusters[@]}"); do
    echo -e "${GREEN}Launching on ${cluster}${NC}"
    oarsub "source ~/.bashrc ; source ~/ReproducibleRL/code/g5k_tools/g5k_env_unfrozen.sh && launch_unfrozen_computations"\
           -l nodes=1,walltime="24:00:00" \
           -p "${cluster}" \
           -q "besteffort"
done

echo "Finished processing all clusters in ${SERVER_NAME}."
