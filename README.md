# ReproducibleRL

In this repository are gathered all the code necessary to reproduce the figures in the paper "Is Deep RL Reproducible? State of the art and new tools". Each experiment is associated with a folder in the `code/` directory. The functions to make the actual plots can be found in `code/plots_scripts/`.

The makefile contains several recipe that can be useful and explain some classical tasks that you may be interested to do in this repository. Among them:

- `make venv_plot` run automatically the 2 next commands (create the venv, then create the plots)
    - `make .venv_plot` create the .venv environment to make the plot
    - `make plot` create the plots used in the article from the data in code/results.
- `make test` run a test of the experiments.
- `make EXPERIMENT_NAME` will execute the experiment `EXPERIMENT_NAME`, for example one can execute `make code/Acrobot/Acrobot_guix` to execute and generate (in the folder `code/results`) the results of the experiment.

This repository also contains the html for a static website with further results in the `public` directory.
