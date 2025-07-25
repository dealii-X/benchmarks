

set terminal pngcairo enhanced font "Arial,14" size 800,600 lw 2

set output "kokkos_intel_pvc.png"


set xlabel "Number of degrees of freedom"
set ylabel "gdofs/second"


dofs_per_elem(nq) = (nq-1)**3

set grid
set border linewidth 0.3
#set format y "10^{%L}"
set format x "10^{%L}"

set yr [1e-1:1e2]
set xr [1e4:1e8]

set log xy
set key top left box lw 0 height 0.5

set title "BK1, 1 GPU tile Data Center GPU Max 1550 (SMNG-P2), Kokkos"
set output "kokkos_intel_pvc.png"

plot "kokkos_intel_pvc_p2.txt" u (($1)*dofs_per_elem(2)):2 w lp t "nq = 2",\
     "kokkos_intel_pvc_p3.txt" u (($1)*dofs_per_elem(3)):2 w lp t "nq = 3",\
     "kokkos_intel_pvc_p4.txt" u (($1)*dofs_per_elem(4)):2 w lp t "nq = 4",\
     "kokkos_intel_pvc_p5.txt" u (($1)*dofs_per_elem(5)):2 w lp t "nq = 5",\
     "kokkos_intel_pvc_p6.txt" u (($1)*dofs_per_elem(6)):2 w lp t "nq = 6",\
     "kokkos_intel_pvc_p7.txt" u (($1)*dofs_per_elem(7)):2 w lp t "nq = 7",\
     "kokkos_intel_pvc_p8.txt" u (($1)*dofs_per_elem(8)):2 w lp t "nq = 8",\
     "kokkos_intel_pvc_p9.txt" u (($1)*dofs_per_elem(9)):2 w lp t "nq = 9"

set title "BK1, 1 GPU tile Data Center GPU Max 1550 (SMNG-P2), OpenCL"

layout = "dynamic"
kernel = "BwdTransHexKernel_QP_1D_3D_BLOCKS_SimpleMap"

set output sprintf("opencl_intel_pvc_%s_%s.png", layout, kernel)

plot for [q=2:9] sprintf("< awk '$1 == \"%s\" && $2 == \"%s\"' intel_pvc_q%d.txt", layout, kernel, q) \
     using (($3)*dofs_per_elem(q)):4 w lp title sprintf("nq = %d", q)
