SHELL=bash

code_path = $(shell pwd)/code

# define all the experiment targets
experiments_env = Acrobot Mujoco Atari
experiments_conda_frozen = $(foreach wrd,$(experiments_env),code/$(wrd)/$(wrd)_conda_frozen)
experiments_conda_unfrozen = $(foreach wrd,$(experiments_env),code/$(wrd)/$(wrd)_conda_unfrozen)
experiments_pip_frozen = $(foreach wrd,$(experiments_env),code/$(wrd)/$(wrd)_pip_frozen)
experiments_pip_unfrozen = $(foreach wrd,$(experiments_env),code/$(wrd)/$(wrd)_pip_unfrozen)
experiments_uv_frozen = $(foreach wrd,$(experiments_env),code/$(wrd)/$(wrd)_uv_frozen)
experiments_uv_unfrozen = $(foreach wrd,$(experiments_env),code/$(wrd)/$(wrd)_uv_unfrozen)
experiments_guix = $(foreach wrd,$(experiments_env),code/$(wrd)/$(wrd)_guix)
experiments_guix_avx = $(foreach wrd,$(experiments_env),code/$(wrd)/$(wrd)_guix_with_masked_AVX)

# All experiments to execute on Grid5k
all_biwise_xp = $(experiments_conda_frozen) $(experiments_conda_unfrozen) $(experiments_pip_frozen) $(experiments_pip_unfrozen) $(experiments_uv_frozen) $(experiments_uv_unfrozen) $(experiments_guix) $(experiments_guix_avx) code/Acrobot_GPU

# Make all phony targets to just run it.
.PHONY: plot $(all_biwise_xp)

venv_plot: .venv_plot plot

.venv_plot: 
	python -m venv .venv_plot && source .venv_plot/bin/activate && python -m pip install -r code/plots_scripts/requirements.txt	

plot:
	[[ -d .venv_plot ]] && source .venv_plot/bin/activate ; bash make_all_plots.sh

waffle:
	cd code/plots_scripts && python waffle_plotly_g5k.py . ../results/results_g5k && mv g5k_website.html ../../public/index.html

test:
	cd code/tests/ && bash tests_small.sh

$(all_biwise_xp): 
	# Make the result folder 
	mkdir -p $(code_path)/results/$(shell realpath $@ --relative-to=code) ;\
	# Execute the experiment
	cd $@ && source setup_environment.sh && launch_my_computation $(code_path)/results/$(shell realpath $@ --relative-to=code)

code/fitted_q:
	mkdir -p code/results/fitted_q
	cd code/fitted_q &&  source setup_environment.sh && launch_my_computation ../results/fitted_q

code/Stability_Study:
	cd code/Stability_Study &&  source setup_environment.sh && launch_my_computation ../results/

code/Mujoco_SAC_PPO/:
	cd code/Mujoco_SAC_PPO/ &&  source setup_environment.sh && launch_my_computation ../results/ "Hopper-v5"


# Local Variables:
# mode: makefile
# End:

