launch_my_computation (){
# set -e

mkdir -p $1
uv run ../run_ppo.py $1 && cp uv.lock ${1}/server_env_explicit.txt
}

echo "Launch function created, you can now run launch_my_computation [output_folder]"
