launch_my_computation (){
# set -e
venv_dir=$(mktemp -d --suffix="reprovenv_$(basename $(realpath .))")
echo "Virtual environment in " $venv_dir
clean(){
    conda remove --quiet --prefix $venv_dir --all --yes &> /dev/null
}

mkdir -p $1

conda create --quiet --prefix $venv_dir --yes --file conda_env_explicit.txt \
&& conda run --prefix $venv_dir python3 ../run_ppo.py $1
clean
}
echo "Launch function created, you can now run launch_my_computation [output_folder]"
