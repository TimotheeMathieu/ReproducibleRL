#/usr/bin/env bash

print_help (){
printf "$0 |[username] [output_folder]: \n\ncopy experiment results from all servers on g5k in output_folder \n\n"
}

if [[ $# -eq 0 ]] ; then
    print_help
    exit 0
fi

servers=(
    "grenoble"
    "lyon"
    "nancy"
    "rennes"
    "sophia"
)

USERNAME=$1
OUTPUT_FOLDER=$2

GREEN='\033[1;32m'
NC='\033[0m'

TEMP_FOLDER=/tmp/repro_result_temp
mkdir $TEMP_FOLDER

for server in "${servers[@]}"; do
    echo -e "${GREEN}Copying from ${server}${NC}"
    scp -r $USERNAME@access.grid5000.fr:~/${server}/ReproducibleRL/code/results $TEMP_FOLDER
done


echo clean results
for env in "Acrobot" "Atari" Mujoco; do
    for venv in "conda_frozen" "conda_unfrozen" "guix" "guix_with_masked_AVX" "pip_frozen" "pip_unfrozen"; do
        for server in $TEMP_FOLDER/results/$env/${env}_$venv/* ; do 
            server_name=$(basename $server)
            if [ -d $TEMP_FOLDER/results/$env/${env}_$venv/$server_name/ ]; then
                current_path=$TEMP_FOLDER/results/$env/${env}_$venv/$server_name/0.monitor.csv
                cat $current_path | tail -n +2 | cut -d "," -f -2 | sponge  $current_path
            fi
        done
    done
done

cp -rf ${TEMP_FOLDER}/results/* ${OUTPUT_FOLDER}/
rm -rf ${TEMP_FOLDER}

echo done!
