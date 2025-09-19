"""
globalRW    = global memory read and write
sharedRW    = shared memory read and write
byte        = size of data-type
dt          = total kernel execution time on GPU (in seconds)
"""

import matplotlib.pyplot as plt
import numpy as np


def AchievedPerformance(nq0, nq1, nq2, nelmt, numBlocks, byte, dt):
    
    globalRW = 0; sharedRW = 0; flop = 0

    #Copy to Shmem
    globalRW += numBlocks * (nq0 * nq0 + nq1 * nq1 + nq2 * nq2)
    sharedRW += numBlocks * (nq0 * nq0 + nq1 * nq1 + nq2 * nq2)

    #Load Geometric Factors
    globalRW += nelmt * nq0 * nq1 * nq2 * 6
    
    # Multiply by D
    sharedRW += nelmt * nq0 * nq1 * nq2 * (nq0 + nq1 + nq2)
    globalRW += nelmt * nq0 * nq1 * nq2 * (nq0 + nq1 + nq2)
    flop     += nelmt * nq0 * nq1 * nq2 * (nq0 + nq1 + nq2) * 2

    # Apply chain rule
    sharedRW += nelmt * nq0 * nq1 * nq2 * 3
    flop     += nelmt * nq0 * nq1 * nq2 * 3 * 5

    # Multiply with DT
    sharedRW += nelmt * nq0 * nq1 * nq2 * (nq0 + nq1 + nq2) * 2
    flop     += nelmt * nq0 * nq1 * nq2 * (nq0 + nq1 + nq2) * 2
    globalRW += nelmt * nq0 * nq1 * nq2


    results = {
        "globalRW"    : globalRW,
        "sharedRW"    : sharedRW,
        "flop"        : flop,
        "flop/s"      : flop / dt,
        "AI_global"   : flop / (globalRW * byte),
        "AI_shared"   : flop / (sharedRW * byte),     
        }
    
    return results


def generate_roofline(rooflines, achieved_points=None, filename="rooflineBK5.pdf"):
    """
    Plots and saves a roofline model from a {bandwidth: tflops} dictionary.

    Parameters:
    - rooflines (dict): {bandwidth_GB/s: peak_TFLOPs}
    - achieved_points (dict): {AI: TFLOPs/s}
    - filename (str): Output PDF filename
    """
    ai_range = np.logspace(-2, 3, 500)
    colors = plt.cm.tab10.colors
    plt.figure(figsize=(10, 6))

    max_perf = max(rooflines.values()) if rooflines else 0
    for i, (bw, tflops) in enumerate(rooflines.items()):
        # Performance (TFLOPs/s) = AI (FLOPs/Byte) * BW (GB/s) / 1000
        perf = np.minimum((bw / 1000) * ai_range, tflops)
        color = colors[i % len(colors)]
        label = f'{bw} GB/s, {tflops} TFLOPs'

        plt.loglog(ai_range, perf, linewidth=3, label=label, color=color)
        plt.axhline(tflops, color=color, linestyle='--', linewidth=1)
        # Crossover AI = Peak Perf (TFLOPs/s) * 1000 / BW (GB/s)
        crossover_ai = tflops * 1000 / bw
        plt.axvline(crossover_ai, color=color, linestyle=':', linewidth=1)

    # Plot achieved points with formatted labels
    if achieved_points:
        for ai, tflops_val in achieved_points.items():
            plt.scatter(ai, tflops_val, color='black', marker='x', s=80, zorder=5)
            label_text = f'AI={ai:.3g}, TFLOP/s={tflops_val:.3g}'
            plt.text(ai * 1.05, tflops_val * 1.1, label_text, fontsize=9, ha='left', va='bottom')
            max_perf = max(max_perf, tflops_val)

    # Set axis limits, starting y-axis lower to show sub-1 TFLOP/s performance
    plt.xlim(1e-2, 1e3)
    plt.ylim(1e-2, 10 ** np.ceil(np.log10(max_perf)))

    # Labels and legend
    plt.xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=12)
    plt.ylabel('Performance (TFLOPs/s)', fontsize=12)
    plt.title('Roofline Model', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, format='pdf')
    plt.close()


def main():
    ###    Input Parameters   ###
    #############################
    nq0 = 4
    nq1 = 4
    nq2 = 4
    nelmt = 524288
    numBlocks = nelmt / 2
    byte = 4
    dt = 0.007
    
    #Global Memory BW
    bandwidth1 = 170
    tflops1 = 5
    
    #Shared Memory BW
    bandwidth2 = 3000
    tflops2 = 5
    ##############################
    
    
    achieved_vals = AchievedPerformance(nq0, nq1, nq2, nelmt, numBlocks, byte, dt)
    print(achieved_vals)

    rooflines = {
        bandwidth1 : tflops1,
        bandwidth2 : tflops2
    }
    achieved_points = {
        achieved_vals["AI_global"] : achieved_vals["flop/s"] * 10**-12,
        achieved_vals["AI_shared"] : achieved_vals["flop/s"] * 10**-12
    }

    generate_roofline(rooflines, achieved_points)
    
if __name__ == "__main__":
    main()






