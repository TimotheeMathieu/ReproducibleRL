launch_my_computation (){
guix time-machine --url=https://codeberg.org/guix/guix.git --unsafe-channel-evaluation --channels=../../channels.scm  -- shell -CNF -L $(realpath ../../guix-rl) python-stable-baselines3 python python-imageio python-pandas python-ale-py python-shimmy python-gym-old bash --share=$(realpath ../..) -- python3 ../run_ppo.py $1
}

echo "Launch function created, you can now run launch_my_computation [output_folder]"
