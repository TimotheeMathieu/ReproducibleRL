import numpy as np
import torch
import torch.nn as nn
import random
import socket
import os
import csv
import sys

torch.use_deterministic_algorithms(True)

n_seeds = 500

device = torch.device("cpu")
results = {"seed":[], "size":[], "value":[]}
for seed in np.arange(n_seeds):
    seed = int(seed)
    for size in [32,64,128]:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)
            return layer
        
        network = nn.Sequential(layer_init(nn.Linear(size, size)),
                                layer_init(nn.Linear(size,1), std=1.0),).to(device)
        
        with torch.no_grad():
            action = network(torch.rand(size=(size,)).to(device))
        results["seed"].append(seed)
        results["size"].append(size)
        results["value"].append(action.cpu().numpy()[0].item())

name = socket.gethostname()

with open(f"{sys.argv[1]}/results_torch_{sys.argv[2]}_{name}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, results.keys())
    w.writeheader()
    for rowid in range(len(results["seed"])):
        w.writerow({k: results[k][rowid] for k in results.keys()})
