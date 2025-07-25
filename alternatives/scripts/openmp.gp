
set terminal pngcairo enhanced
set output "intel_pvc_openmp.png"

set xlabel "Degrees of Freedom"
set ylabel "Throughput (GDoF/s)"

set log x

set format x "10^{%L}"

set title "BK1 (OpenMP), Intel PVC"

plot "intel_pvc_openmp_subdevice.txt" u 2:3 w lp t "SUBDEVICE",\
     "intel_pvc_openmp_device.txt" u 2:3 w lp t "DEVICE"