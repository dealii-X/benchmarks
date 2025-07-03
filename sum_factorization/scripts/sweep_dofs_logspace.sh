#!/bin/bash

exec_name="./opencl_benchmark"
nq0=4
nq1=4
nq2=4

nm0=$((nq0 - 1))
nm1=$((nq1 - 1))
nm2=$((nq2 - 1))

vol_per_elem=$((nm0 * nm1 * nm2))

# Generate 21 log-spaced ndofs values from 1e4 to 1e7 using Python
ndofs_list=$(python3 -c "import numpy as np; print(' '.join(str(int(x)) for x in np.logspace(4, 7, 21)))")

for ndofs in $ndofs_list; do
    nelmt=$((ndofs / vol_per_elem))
    echo "Running: ndofs=$ndofs -> nelmt=$nelmt"
    $exec_name $nq0 $nq1 $nq2 $nelmt
done
