
launch_guix_pip_computations(){
GREEN='\033[1;32m'
NC='\033[0m'
HOSTNAME=$(cat /etc/hostname | cut -d '-' -f 1)
ROOT="${HOME}/ReproducibleRL/code" # root of the code.
module load conda 

mkdir -p ${ROOT}/results/Minatar/MinatarBreakout_MLP/
mkdir -p ${ROOT}/results/Minatar/MinatarBreakout_CNN/
source ${ROOT}/Minatar/setup_environment.sh 
if cd ${ROOT}/Minatar/ && launch_my_computation tmp/Minatar/ ; then
    mv ${ROOT}/Minatar/tmp/Minatar/MinatarBreakout_MLP/${HOSTNAME}* ${ROOT}/results/Minatar/MinatarBreakout_MLP/${HOSTNAME}
    mv ${ROOT}/Minatar/tmp/Minatar/MinatarBreakout_CNN/${HOSTNAME}* ${ROOT}/results/Minatar/MinatarBreakout_CNN/${HOSTNAME}
    echo "Moved minatar results"
fi

bash ${ROOT}/tools/dump_metadata.sh ~/ReproducibleRL/code/results/metadata/
}
