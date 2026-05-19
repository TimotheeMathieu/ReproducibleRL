launch_my_computation (){
# set -e
venv_dir=$(mktemp -d --suffix="reprovenv_$(basename $(realpath .))")
echo "Virtual environment in " $venv_dir
clean(){
    [[ -d $venv_dir ]] && rm -r $venv_dir
    [[ "$VIRTUAL_ENV" != "" ]] && deactivate
}

mkdir -p $1

python3 -m venv $venv_dir \
&& source ${venv_dir}/bin/activate \
&& python3 -m pip install --upgrade pip \
&& python3 -m pip install -r requirements.txt \
&& python3 ../run_ppo.py $1 \
&& pip freeze --all > ${1}/server_env_explicit.txt
clean
}

echo "Launch function created, you can now run launch_my_computation [output_folder]"
