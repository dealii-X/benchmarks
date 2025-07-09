import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

filename = "nvidia_a100_filtered.txt"
title = "Nvidia A100"

# Updated column names
colnames = ["kernel", "nelmt", "GDoF_per_s"]

# Read the data
df = pd.read_csv(filename, sep=r"\s+", header=None, names=colnames)

# Compute degrees of freedom per element
nq0, nq1, nq2 = 4, 4, 4
nm0, nm1, nm2 = nq0 - 1, nq1 - 1, nq2 - 1
dofs_per_elem = nm0 * nm1 * nm2
df["dofs"] = df["nelmt"] * dofs_per_elem

# Plotting
plt.figure(figsize=(10, 6))

for kernel, group in df.groupby("kernel"):
    plt.plot(group["dofs"], group["GDoF_per_s"], marker='o', label=kernel)

plt.xscale("log")
plt.xlabel("Degrees of Freedom (DoFs)")
plt.ylabel("Throughput (GDoF/s)")
plt.title(title + f"\nnq0 = nq1 = nq2 = {nq0}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("perf_vs_dofs.png", dpi=150)
plt.show()
