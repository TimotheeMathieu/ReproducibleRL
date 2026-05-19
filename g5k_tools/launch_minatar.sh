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
mkdir -p ~/ReproducibleRL/code/results/metadata/

GREEN='\033[1;32m'
NC='\033[0m'

if [ -d $HOME/.pyenv ] ; then 
    echo "Using installed pyenv, I hope that 3.12.12 is installed"
else
    curl -fsSL https://pyenv.run | bash
    export PYENV_ROOT="$HOME/.pyenv"
    [[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init - bash)"
    pyenv install -s 3.12.12
fi

set +e # allow fail, e.g. for exotic jobs for instance

# Loop over each cluster
for cluster in $(echo "${clusters[@]}"); do
    walltime="12:00:00"
    queue="besteffort"

    echo -e "${GREEN}Launching on ${cluster}${NC}"
    oarsub "module --ignore_cache load conda ; source ~/ReproducibleRL/code/g5k_tools/g5k_env_minatar.sh && PATH=~/.pyenv/versions/3.12.12/bin:$PATH launch_guix_pip_computations"\
           -l nodes=1,walltime=$walltime \
           -p "${cluster}" \
           -q $queue

done

echo "Finished processing all clusters in ${SERVER_NAME}."
