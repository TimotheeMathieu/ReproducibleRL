
launch_my_computation (){
clean(){
    [[ -d /tmp/venv_reproductibility ]] && rm -r /tmp/venv_reproductibility
    [[ "$VIRTUAL_ENV" != "" ]] && deactivate
}

mkdir -p $1

# We need the environment variable to have deterministic enabled in torch with gpu.
python3 -m venv /tmp/venv_reproductibility && source /tmp/venv_reproductibility/bin/activate && python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt && CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 run_ppo.py $1 
clean
}

echo "Launch function created, you can now run launch_my_computation [output_folder]"
