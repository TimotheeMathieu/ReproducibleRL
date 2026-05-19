import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns

data_burns = np.loadtxt("../results/mujoco_sac_vs_ppo/Hopper-v5_ppo_logs_burns_99.csv")
data_flanders = np.loadtxt("../results/mujoco_sac_vs_ppo/Hopper-v5_ppo_logs_flanders_99.csv")




df = pd.DataFrame()
for n in range(1,100):
    ci1 = stats.bootstrap((data_burns[:(n+1)],), np.mean)
    df = pd.concat([df, pd.DataFrame({"n":[n+1],
                                      "ci_width":[ci1.confidence_interval.high-ci1.confidence_interval.low],
                                      "name":["burns"]})], ignore_index=True)

    ci2 = stats.bootstrap((data_flanders[:(n+1)],), np.mean)
    df = pd.concat([df, pd.DataFrame({"n":[n+1],
                                      "ci_width":[ci2.confidence_interval.high-ci2.confidence_interval.low],
                                      "name":["flanders"]})], ignore_index=True)

fig, ax = plt.subplots()
sns.lineplot(df,  x="n", y="ci_width", hue="name",ax = ax)

plt.show()


df = pd.DataFrame()
for n in range(1,100):
    ci1 = stats.bootstrap((data_burns[:(n+1)],), np.mean)
    df = pd.concat([df, pd.DataFrame({"n":[n+1],
                                      "lb":[ci1.confidence_interval.low],
                                      "ub":[ci1.confidence_interval.high],
                                      "name":["burns"]})], ignore_index=True)

    ci2 = stats.bootstrap((data_flanders[:(n+1)],), np.mean)
    df = pd.concat([df, pd.DataFrame({"n":[n+1],
                                      "lb":[ci2.confidence_interval.low],
                                      "ub":[ci2.confidence_interval.high],
                                      "name":["flanders"]})], ignore_index=True)

fig, ax = plt.subplots()
sns.lineplot(df,  x="n", y="lb", hue="name",ax = ax)
sns.lineplot(df,  x="n", y="ub", hue="name",ax = ax)
plt.show()
