#!/bin/bash

libraries="seaborn statsmodels pandas matplotlib scipy distinctipy matplotlib_set_diagrams"
test_libs_python (){
    errorc=0

    for lib in $libraries ; do
        python -c "import ${lib}" &>/dev/null
        if [ $? -ne 0 ] ; then
          printf "%s is not installed. \n" "$lib"
          errorc=1
        fi   

    done
}
error_python_lib=$(test_libs_python)

if [[ $error_python_lib ]] ; then
    echo $error_python_lib
    exit 1
else
    echo "Plots will be created in the Figures directory"
    mkdir -p papers/Figures
    cd code/plots_scripts/
    python plot_stats_mujoco.py ../../papers/Figures
    python plot_stability.py ../results/ ../../papers/Figures
    python plot_cost.py ../../papers/Figures ../results
    python make_g5k_bars.py ../../papers/Figures ../results/results_g5k
    python plot_g5k.py ../../papers/Figures ../results/results_g5k
    python plot_histogram_torch_avx.py ../../papers/Figures ../results/
    python process_ppo_focus.py ../../papers/Figures
    python plot_policy_trajectory.py plot ../../papers/Figures
    python process_policy_divergence.py plot ../../papers/Figures --metric tv
    cd ../../census
    python euler_diagram.py ../papers/Figures
fi

