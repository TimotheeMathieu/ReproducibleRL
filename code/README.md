# How to

The directory  `g5k_tools` contains helper scripts for the server grid 5000. The directory `tools` contains python script used collect data during training. The directory `plots_scripts` contains scripts and hardware data used to make the plots. The directory `results` contains the results of our experiments. 

All the other directory are experiment directory, they contain everything to reproduce an experiment. In particular, each experiment directory contains a `setup_environment.sh` script which, once sourced, define a `launch_my_computation` cli command which  can be used to launch the computation.

All the experiments were done either on computers from grid 5000 or on two local servers Flanders and Burns. The description of the hardware and software on these computers is given in the folder `result/metadata` using the hostname of a given server.

# Launch experiments locally

## prerequisites
You need to install `conda` and  `guix` before running the experiments locally. You can find them here :  
Conda : https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html  
Guix : https://guix.gnu.org/manual/1.5.0/en/guix.html#Installation

## Run the experiment
To run locally, you can use the function written in setup_environment.sh in each directory which define a bash function launch_my_computation:

```
cd EXPERIMENT_FOLDER
source setup_environment.sh
launch_my_computation output_folder
```

This will generate results in output_folder. Then these results can be processed using ../plots/process_pickles.py script to get a csv with the results

### Reproducible Atari local
To test the local reproducibility on Atari breakout, you can run :
```
cd xp_atari_local_repro/
source run_xp.sh
```
If needed, you will find the full results in `/results/xp_atari_local_repro/`

# Launch on g5k

We also give a few tools to launch experiments of grid5000 servers. For example, to execute the experiments on the `nancy` servers if your username is `bob`, do

```
bash g5k_tools/copy_scripts_to_servers.sh bob
ssh bob@access.grid5000.fr
ssh nancy
cd ReproducibleRL/code/g5k_tools/
bash launch_on_server.sh nancy
bash launch_unfrozen.sh nancy
exit
```

You can redo this on the other servers and then, once the computations are done, you can copy back the results with 

```
mkdir results/
bash g5k_tools/copy_results_from_servers.sh bob results
```

which will result in a `results` folder containing the results.


# Process the results

The results can be processed into the plots from the article with scripts from the `plots_scripts` directory. See the Readme in the `plots_scripts` directory.
