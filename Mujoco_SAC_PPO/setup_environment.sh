
launch_ppo_computation (){
venv_dir=$(mktemp -d --suffix="reprovenv")
echo "Virtual environment in " $venv_dir
clean(){
    [[ -d $venv_dir ]] && rm -r $venv_dir
    [[ "$VIRTUAL_ENV" != "" ]] && deactivate
}

mkdir -p $1

# Use the requirements.txt exported from nancy computer at the time to make the computation
# Run ppo xps
python3 -m venv $venv_dir && source ${venv_dir}/bin/activate && python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt && python3 run_ppo.py $1 --environment $2 
python3 -m venv $venv_dir && source ${venv_dir}/bin/activate && python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt && python3 run_ppo.py $1 --environment $2 --pretrained

# Run sac xps
python3 -m venv $venv_dir && source ${venv_dir}/bin/activate && python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt && python3 run_sac.py $1 --environment $2
python3 -m venv $venv_dir && source ${venv_dir}/bin/activate && python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt && python3 run_sac.py $1 --environment $2 --pretrained
clean
}

echo "Launch function created, you can now run launch_my_computation [output_folder] [env-name]"
