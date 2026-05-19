
launch_my_computation (){
if [[ -z $(lscpu | grep avx2) ]] ; then
    echo "Computing without avx mask"
    guix time-machine --url=https://codeberg.org/guix/guix.git --unsafe-channel-evaluation --channels=../../channels.scm -- shell -CNF  python-stable-baselines3 python python-imageio python-pandas bash --share=$(realpath ../..) -- python3 ../run_ppo.py $1
elif [[ -z $(lscpu | grep avx512) ]] ; then
    echo "Computing with avx2 mask"
    guix time-machine --url=https://codeberg.org/guix/guix.git --unsafe-channel-evaluation --channels=../../channels.scm -- shell -CNF --with-graft=glibc=gcvi2 gcvi2  python-stable-baselines3 python python-imageio python-pandas bash --share=$(realpath ../..) -- python3 ../run_ppo.py $1
else
    echo "Computing with avx2 and 512 mask"
    guix time-machine --url=https://codeberg.org/guix/guix.git --unsafe-channel-evaluation --channels=../../channels.scm -- shell --with-graft=glibc=gcvir gcvir --max-jobs=1 --cores=1 -CNF python-stable-baselines3 python python-imageio python-pandas bash --share=$(realpath ../..) -- python3 ../run_ppo.py $1
fi
}

echo "Launch function created, you can now run launch_my_computation [output_folder]"
