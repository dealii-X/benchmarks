#!/bin/bash

# Get polynomial order (default = 4 if not provided)
nq=${1:-4}

exec_name="./BK1/kokkos_benchmark"
nq0=$nq
nq1=$nq
nq2=$nq

nm0=$((nq0 - 1))
nm1=$((nq1 - 1))
nm2=$((nq2 - 1))

vol_per_elem=$((nm0 * nm1 * nm2))

# Set OpenMP environment variables
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Generate 21 log-spaced ndofs values from 1e4 to 1e8 using Python
ndofs_list=$(python3 -c "import numpy as np; print(' '.join(str(int(x)) for x in np.logspace(4, 8, 21)))")

for ndofs in $ndofs_list; do
    nelmt=$((ndofs / vol_per_elem))
    $exec_name $nq0 $nq1 $nq2 $nelmt | awk '/Kokkos/ { print $5, $8 }'
done