
pip_non_deterministic() {
    [[ -d my_venv ]] && \rm -r my_venv --force
    python3 -m venv my_venv 
    source my_venv/bin/activate  && python3 -m pip install --upgrade pip  && python3 -m pip install -r requirements.txt && python3 ./run_ppo.py  --deterministic $1 $2 && deactivate
    \rm -r my_venv --force
}


pip_deterministic() {
    [[ -d my_venv ]] && \rm -r my_venv --force
    python3 -m venv my_venv 
    source my_venv/bin/activate  && python3 -m pip install --upgrade pip  && python3 -m pip install -r requirements.txt && python3 ./run_ppo.py  --deterministic $1 $2 && deactivate
    \rm -r my_venv --force
}

conda_deterministic() {
    # conda create reprorl python=3.12
    conda activate reprorl
    # conda install conda-forge::gymnasium conda-forge::stable-baselines3 conda-forge::gymnasium-mujoco
    python ./run_ppo.py --deterministic $1 $2
    conda deactivate
}

guix_with_avx() {
    guix time-machine --url=https://codeberg.org/guix/guix.git --channels=../channels.scm -- shell -CNF python-stable-baselines3 python-gymnasium-next python-mujoco python python-imageio python-pandas bash --share=$(realpath ..) -- python3 ./run_ppo.py --deterministic $1 $2
}


guix_without_avx512() {
    guix time-machine --url=https://codeberg.org/guix/guix.git --channels=../channels.scm -- shell --with-graft=glibc=gcvi5 gcvi5 --max-jobs=1 --cores=1 -CNF python-stable-baselines3 python python-gymnasium-next python-mujoco python-imageio python-pandas bash --share=$(realpath ..) -- python3 ./run_ppo.py --deterministic $1 $2
}

guix_without_avx2() {
    guix time-machine --url=https://codeberg.org/guix/guix.git --channels=../channels.scm -- shell --with-graft=glibc=gcvi2 gcvi2 --max-jobs=1 --cores=1 -CNF python-stable-baselines3 python python-imageio python-gymnasium-next python-mujoco  python-pandas bash --share=$(realpath ..) -- python3 ./run_ppo.py --deterministic $1 $2
}

get_time (){
case "$1" in
    "vanilla") pip_non_deterministic $1 $2;;
    "deterministic") pip_deterministic $1 $2;;
    "conda")  conda_deterministic $1 $2;;
    "guix")  guix_with_avx $1 $2;;
    "guix-avx512")  guix_without_avx512 $1 $2;;
    "guix-avx2")  guix_without_avx2 $1 $2;;
esac
}

n_rep=10

launch_my_computation (){
    for workflow in "guix" "conda" "guix-avx2"; do
        >&2 echo "Doing computation for $workflow" 
        for f in $(seq $n_rep); do
            get_time $workflow $1
        done
    done
}

echo "Launch function created, you can now run launch_my_computation [output folder]."
