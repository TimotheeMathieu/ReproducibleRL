
launch_guix_pip_computations(){
GREEN='\033[1;32m'
NC='\033[0m'
HOSTNAME=$(cat /etc/hostname | cut -d '-' -f 1)
ROOT="${HOME}/ReproducibleRL/code" # root of the code.
for experiment in "Acrobot" "Atari" "Mujoco"; do
    for env_type in "guix" "guix_with_masked_AVX" "pip_frozen"; do 
        echo -e "${GREEN}Launching experiment $experiment with $env_type ${NC}"
        mkdir -p ${ROOT}/tmp/${experiment}/${experiment}_${env_type}
        mkdir -p ${ROOT}/results/${experiment}/${experiment}_${env_type}

        source ${ROOT}/${experiment}/${experiment}_${env_type}/setup_environment.sh 
        temp_path="${ROOT}/tmp/${experiment}/${experiment}_${env_type}/*/${HOSTNAME}"
        result_path=${ROOT}/results/${experiment}/${experiment}_${env_type}/$HOSTNAME
        if [ -d "$temp_path" ]; then
            rm -rf $temp_path
        fi
        echo ${result_path}
        if [ ! -d "$result_path" ]; then
            if cd ${ROOT}/${experiment}/${experiment}_${env_type} && launch_my_computation ${ROOT}/tmp/${experiment}/${experiment}_${env_type} ; then
                mv ${ROOT}/tmp/${experiment}/${experiment}_${env_type}/*/${HOSTNAME}* $result_path
                echo "Moved results to ~/ReproducibleRL/code/results/${experiment}/${experiment}_${env_type}"
            fi
        fi
    done
    echo -e "Experiment $experiment done"
done

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

launch_conda_computations(){
GREEN='\033[1;32m'
NC='\033[0m'
HOSTNAME=$(cat /etc/hostname | cut -d '-' -f 1)
ROOT="${HOME}/ReproducibleRL/code" # root of the code.
for experiment in "Acrobot" "Atari" "Mujoco"; do
    for env_type in "conda_frozen"; do 
        echo -e "${GREEN}Launching experiment $experiment with $env_type ${NC}"
        mkdir -p ${ROOT}/tmp/${experiment}/${experiment}_${env_type}
        mkdir -p ${ROOT}/results/${experiment}/${experiment}_${env_type}

        source ${ROOT}/${experiment}/${experiment}_${env_type}/setup_environment.sh  
        temp_path=${ROOT}/${experiment}/${experiment}_${env_type}/$HOSTNAME
        result_path=${ROOT}/results/${experiment}/${experiment}_${env_type}/$HOSTNAME
        if [ -d "$temp_path" ]; then
            rm -rf $temp_path
        fi
        if [ ! -d "$result_path" ]; then            
            if cd ${ROOT}/${experiment}/${experiment}_${env_type} && launch_my_computation ${ROOT}/tmp/${experiment}/${experiment}_${env_type} ; then
                mv ${ROOT}/tmp/${experiment}/${experiment}_${env_type}/*/${HOSTNAME}* $result_path
                cp ${ROOT}/tmp/${experiment}/${experiment}_${env_type}/server_env_explicit.txt ${result_path}/
                echo "Moved results to ${ROOT}/results/${experiment}/${experiment}_${env_type}"
            fi        
        fi
    done
    echo -e "Experiment $experiment done"
    bash ${ROOT}/tools/dump_metadata.sh ~/ReproducibleRL/code/results/metadata/
done
}


    

