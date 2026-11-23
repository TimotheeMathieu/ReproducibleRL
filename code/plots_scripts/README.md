The reproduce the figures from the article, you can use first create a virtual environment with the `requirements.txt` file, then run the following commands

```
mkdir Figures
python plot_stats_mujoco.py Figures
python plot_stability.py ../results/ Figures
python plot_cost.py Figures ../results
python make_g5k_bars.py Figures ../results/results_g5k
python plot_g5k.py Figures ../results/results_g5k
python plot_histogram_torch_avx.py Figures ../results/
python process_ppo_focus.py Figures
python plot_policy_trajectory.py plot Figures
python process_policy_divergence.py plot Figures --metric tv kl_fb kl_bf --normalized
```

the resulting plots should then appear in the Figures repository.