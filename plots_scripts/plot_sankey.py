import pandas as pd
from sankey import sankey
import numpy as np
import sys

df = pd.read_csv("../../census/census_rlc.csv", index_col = 0)

def classer(x):
    if x.strip() in ["only-pseudocode", "none", "broken"]:
        return "no working code"
    if x.strip() in ["docker", "conda"]:
        return "fix python and python lib"
    if x.strip() in ["requirements-freeze", "poetry", "requirements-freeze+conda"]:
        return "fix python lib"
    else:
        return "unversionned libs"

df["broad repro"] = df["repro"].apply(lambda x : classer(x))

pd.options.display.max_rows = 8

sankey(
    df['repro'], df["broad repro"], aspect=20, 
    fontsize=12, figure_name=f"{sys.argv[1]}/Reproducibility at RLC"
)

df = pd.read_csv("../../census/papier_icml_iclr.csv", sep = ";")
# df = df.loc[df["a vérifier"] == "x", ]
df = df.drop(["Unnamed: 15", "a vérifier"], axis="columns")

df2 = pd.DataFrame()

def get_tool(stat):
    if stat['Docker/Apptainer'] == "1":
        return "container"
    elif stat['Poetry ou pyproject.toml (0/1)'] == "1":
        return "poetry"
    elif stat['Conda / environment.yml (0/1)'] == "1":
        return "conda"
    elif stat['pip install / setup.py (0/1)'] == "1":
        return "pip"
    elif stat["Conda → pip (0/1)"] == "1":
        return "pip"
    elif stat['pypi'] == "1":
        return "pip"
    else:
        return "none"

def get_repro(stat):
    if stat['Conda / environment.yml (0/1)'] == "1":
        return "Freeze libs & python"
    elif (stat['fix numpy (0/1/na)'] == "1") or (stat['fix_torch (0/1/na)'] == "1"):
        return "Freeze libs"
    elif stat['Poetry ou pyproject.toml (0/1)'] == "1":
        return "Freeze libs"
    elif stat['Docker/Apptainer'] == "1":
        return "Freeze libs & python"
    elif get_tool(stat) == "none":
        return "No code"
    else:
        return "Do not freeze libs"

for i in range(len(df)):
    df2 = pd.concat([df2, 
                     pd.DataFrame({"tool":[get_tool(df.iloc[i])],
                                   "repro":[get_repro(df.iloc[i])]
                                   })], 
                    ignore_index= True)


pd.options.display.max_rows = 8

sankey(
    df2['tool'], df2["repro"], aspect=20, 
    fontsize=12, figure_name= f"{sys.argv[1]}/Reproducibility at ICML & ICLR"
)
