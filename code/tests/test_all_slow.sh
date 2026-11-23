# Suppose that pyenv is already installed on your system, with python 3.12.12 and that ReproducibleRL is not in home directory

set -e 

if [ -d ~/ReproducibleRL ] ;  then  
    echo "~/ReproducibleRL already exists, not cloning"
else
    git clone https://gitlab.inria.fr/scool/ReproducibleRL ~/ReproducibleRL
fi

# Get some colors :)
GREEN='\033[1;32m'
RED='\033[0;31m'
NC='\033[0m'

module() {
    echo "asked to load module $@"
}

oarsub() {
    eval $1
    status=$?
    if [ $status -eq 0  ]; then
        echo -e "${GREEN} $1 passed ! ${NC}"
    else
        echo -e "${RED} $1 failed :( ! ${NC}"
        exit 1
    fi
}


export -f module
export -f oarsub

cd ~/ReproducibleRL/code/g5k_tools && bash ../g5k_tools/launch_on_server.sh test_server
