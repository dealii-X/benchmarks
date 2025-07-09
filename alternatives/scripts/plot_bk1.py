import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#filename="intel_pvc_gpu_O3.txt"
#title = "Intel Data Center GPU"

#filename="apple_m2.txt"
#title="Apple M2 Pro"


filename="nvidia_gh200.txt"
title="Nvidia GH200"

# Define your column names manually
colnames = ["mode", "kernel", "nelmt", "GDoF_per_s"]

# Load the file with no header but assign these column names
df = pd.read_csv(filename, sep="\t", header=None, names=colnames)


# Compute number of DoFs (optional if you want to scale by element volume)
nq0, nq1, nq2 = 4, 4, 4
nm0, nm1, nm2 = nq0 - 1, nq1 - 1, nq2 - 1
dofs_per_elem = nm0 * nm1 * nm2
df["dofs"] = df["nelmt"] * dofs_per_elem

# Set up plot
plt.figure(figsize=(10, 6))

# Plot one line per (mode, kernel)
for (mode, kernel), group in df.groupby(["mode", "kernel"]):
    plt.plot(group["dofs"], group["GDoF_per_s"], marker='o', label=f"{mode}-{kernel}")

plt.xscale("log")
plt.xlabel("Degrees of Freedom (DoFs)")
plt.ylabel("Throughput (GDoF/s)")
plt.title(title + f"\nnq0 = nq1 = nq2 = {nq0}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("perf_vs_dofs.png", dpi=150)
plt.show()
