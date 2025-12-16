#!/usr/bin/env python3
"""
Process benchmark data, plot DOF vs. GDOF/s, and export results.

- Reads a whitespace-separated text file containing repeated headers and data.
- Extracts unique kernels and parameter values.
- Generates per-kernel plots (PDF) and data tables (TSV).
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import sys


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process benchmark data and generate per-kernel plots."
    )
    parser.add_argument("filename", help="Input data file")
    return parser.parse_args()


def load_data(filename: str) -> pd.DataFrame:
    """Load and clean data from file, skipping repeated headers."""
    data = []
    header = None
    seen_header = False

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines

            # Detect header line
            if line.startswith("Kernel"):
                if not seen_header:
                    header = line.split()
                    seen_header = True
                continue  # skip all header lines

            # Process data lines
            parts = line.split()
            if header and len(parts) == len(header):
                data.append(parts)

    if not data or header is None:
        sys.exit("Error: No valid data found in file.")

    # Build DataFrame
    df = pd.DataFrame(data, columns=header)

    # Convert numeric columns
    for col in ["p0", "DOF", "GDOF/s"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.set_index("Kernel", inplace=True)
    return df


def plot_and_export(df: pd.DataFrame):
    """Generate plots and export per-kernel data tables."""
    kernel_list = df.index.unique().to_list()
    p0_values = sorted(df["p0"].dropna().unique().tolist())

    for kernel in kernel_list:
        subset = df.loc[[kernel]]

        plt.figure(figsize=(8, 6))

        for p0_val in p0_values:
            rows = subset[subset["p0"] == p0_val]

            DOFs = rows["DOF"].dropna()
            GDOFs = rows["GDOF/s"].dropna()

            if not DOFs.empty and not GDOFs.empty:
                plt.scatter(DOFs, GDOFs, label=f"p={p0_val}")

        plt.xlabel("Degrees of Freedom (DOF)")
        plt.ylabel("GDOFs per Second (GDOF/s)")
        plt.title(f"Kernel: {kernel}")
        plt.xscale("log")
        plt.legend(loc="upper left")
        plt.grid(True, alpha=0.5)

        file_name = f"{kernel}.png"
        plt.savefig(file_name, dpi=600, bbox_inches="tight")
        plt.close()

        # Export data subset
        out_df = subset.reset_index()[["p0", "DOF", "GDOF/s"]]
        out_df.to_csv(f"{kernel}.tsv", sep="\t", index=False)


def main():
    args = parse_args()
    df = load_data(args.filename)
    plot_and_export(df)


if __name__ == "__main__":
    main()
