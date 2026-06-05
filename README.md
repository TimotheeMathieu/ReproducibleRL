This repository is a companion to the paper "Is Deep RL Reproducible? State of the art and new tools" by Timothée Mathieu, Juliette Achddou, Alex Davey, Hector Kohler , Philippe Preux and Julien Teigny. Please read the paper for more informations on the experiments.

# How to

The directory  `g5k_tools` contains helper scripts for the server grid 5000. The directory `tools` contains python script used collect data during training. The directory `plots_scripts` contains scripts and hardware data used to make the plots. The directory `results` contains the results of our experiments. 

All the other directory are experiment directory, they contain everything to reproduce an experiment. In particular, each experiment directory contains a `setup_environment.sh` script which, once sourced, define a `launch_my_computation` cli command which  can be used to launch the computation.


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

# Launch on g5k

We also give a few tools to launch experiments of grid5000 servers. For example, to execute the experiments on the `nancy` servers if your username is `bob`, do

```
bash g5k_tools/copy_scripts_to_servers.sh bob
ssh bob@access.grid5000.fr
ssh nancy
cd ReproducibleRL/code/g5k_tools/
bash launch_on_server.sh nancy
exit
```

You can redo this on the other servers and then, once the computations are done, you can copy back the results with 

```
mkdir results/
bash g5k_tools/copy_results_from_servers.sh bob results
```

which will result in a `results` folder containing the results.


# Process the results

We can process the results using the python script in plots_scripts. The script needs as input the folder with the data and as second argument an output folder.
```
python3 plots_scripts/process_pickles.py results/Acrobot/Acrobot_guix_with_masked_AVX results/Acrobot
```

Once all the results have been copied to `results`, these can the be used to reproduce the various plots of the article.
