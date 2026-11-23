import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from matplotlib_set_diagrams import EulerDiagram
import sys
import os

assert len(sys.argv) > 1,  "Output folder needed as argument"

"""
Create an Euler diagram to show the different workflows used in the selected papers from ICLR 2025 and ICML 2025.
For each paper, we check if the code is available and, if so, whether the Python version is fixed, whether the Python libraries are frozen, and whether the system libraries are frozen.
"""
print("--- starting census/euler_diagram.py : generating Euler diagram for the selected papers from ICLR 2025 and ICML 2025 ---")

df = pd.read_csv("paper_survey.csv")
df["Python version frozen"] = df["Python_version_frozen"] == "Yes"
df["Python lib frozen"] = df["Python_lib_frozen"] == "Yes"
df["System frozen"] = [ x not in ["No", np.nan] for x in df["System_frozen"]]
df["Have code"] = df["Code"] == "Yes"
df["pseudocode"] = [ "Pseudo-code" in str(x) for x in df["URL_Git"]]
fig, ax = plt.subplots(figsize=(7,7))

set_colors = plt.get_cmap("tab10")

def get_length(df,cats, all_cats):
    mask = np.ones(len(df))==1
    for cat in all_cats:
        if cat in cats:
            mask = mask & df[cat]
        else:
            mask = mask & (~df[cat])
    return int(np.sum(mask))


cats = ["Python version frozen","Python lib frozen","System frozen",  "Have code"]
sets = list(itertools.product([0,1], repeat=len(cats)))[1:]

dicset = { tuple(s) : get_length(df,[ cats[i] for i in range(len(cats))  if s[i] == 1], cats) for s in sets }
# print(dicset)
diagram = EulerDiagram(
    dicset,
    cost_function_objective="logarithmic", 
    set_labels=cats,
    ax=ax)


for ii, text in enumerate(diagram.set_label_artists):
    text.set_color(set_colors(ii))
    text.set_x(text.get_position()[0]-0.2)
    if ii == 1:
        text.set_y(text.get_position()[1]+0.3)
        text.set_x(text.get_position()[0]+2.5)
    if ii == 0:
        text.set_y(text.get_position()[1]+0.5)
        text.set_x(text.get_position()[0]+4)


output_path = os.path.join(sys.argv[1], "euler_census.jpg") 
plt.savefig(output_path, bbox_inches="tight", dpi=500)
print("saved figure to : ", output_path)
