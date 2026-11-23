launch_my_computation (){
# set -e
venv_dir=$(mktemp -d --suffix="reprovenv_$(basename $(realpath .))")
echo "Virtual environment in " $venv_dir
clean(){
    conda remove --quiet --prefix $venv_dir --all --yes &> /dev/null
}

mkdir -p $1

conda env create --quiet --prefix $venv_dir --file conda_env.yml && conda run --prefix $venv_dir python3 ../run_ppo.py $1 && conda list --prefix $venv_dir --explicit > ${1}/server_env_explicit.txt
clean
}
echo "Launch function created, you can now run launch_my_computation [output_folder]"
