launch_unfrozen_computations(){
GREEN='\033[1;32m'
NC='\033[0m'
HOSTNAME=$(cat /etc/hostname | cut -d '-' -f 1)
ROOT="${HOME}/ReproducibleRL/code" # root of the code.
for experiment in "Acrobot" "Mujoco" "Atari"; do
    for env_type in "pip_unfrozen" "conda_unfrozen"; do 
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
        if cd ${ROOT}/${experiment}/${experiment}_${env_type} && launch_my_computation ${ROOT}/tmp/${experiment}/${experiment}_${env_type} ; then
            mv ${ROOT}/tmp/${experiment}/${experiment}_${env_type}/*/${HOSTNAME}* $result_path
            cp ${ROOT}/tmp/${experiment}/${experiment}_${env_type}/server_env_explicit.txt ${result_path}/
            echo "Moved results to ~/ReproducibleRL/code/results/${experiment}/${experiment}_${env_type}"
        fi
    done
    echo -e "Experiment $experiment done"
done
}


    

