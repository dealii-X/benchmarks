########################################
# INPUT
########################################

mytitle = "BK1 on 1x<GPU_NAME>"

polyOrderBegin = 1
polyOrderEnd   = 8
enableSVG = 0

########################################
# SETUP
########################################

set title mytitle
set xlabel "Degrees of Freedom (DOF)"
set ylabel "Throughput (GDOF/s)"
set logscale x
set format x "10^{%L}"
set grid
set autoscale
set key top left
set offsets graph 0.05, graph 0.05, graph 0, graph 0
set pointsize 1.8


########################################
# PLOT: kokkos (PNG)
########################################

set terminal pngcairo size 1200,800 enhanced font 'Verdana,12'
set output 'kokkos.png'

plot for [i = polyOrderBegin : polyOrderEnd] \
     'kokkos.txt' using 1:(column(i+2 - polyOrderBegin)) \
     with points pointtype (i - polyOrderBegin + 1) \
     title sprintf("p = %d", i)

unset output

########################################
# PLOT: kokkos (SVG)
########################################
if (enableSVG){
     set terminal svg size 1200,800 enhanced font 'Verdana,12' 
     set output 'kokkos.svg' 
     plot for [i = polyOrderBegin : polyOrderEnd] \
     'kokkos.txt' using 1:(column(i+2 - polyOrderBegin)) \
     with points pointtype (i - polyOrderBegin + 1) \
     title sprintf("p = %d", i)
}

unset output

#######################################
# PLOT: cudaWarp (PNG)
########################################

set terminal pngcairo size 1200,800 enhanced font 'Verdana,12'
set output 'cudaWarp.png'

plot for [i = polyOrderBegin : polyOrderEnd] \
     'cudaWarp.txt' using 1:(column(i+2 - polyOrderBegin)) \
     with points pointtype (i - polyOrderBegin + 1) \
     title sprintf("p = %d", i)

unset output

########################################
# PLOT: cudaWarp (SVG)
########################################
if (enableSVG){
     set terminal svg size 1200,800 enhanced font 'Verdana,12' 
     set output 'cudaWarp.svg' 
     plot for [i = polyOrderBegin : polyOrderEnd] \
     'cudaWarp.txt' using 1:(column(i+2 - polyOrderBegin)) \
     with points pointtype (i - polyOrderBegin + 1) \
     title sprintf("p = %d", i)
}

unset output

#######################################
# PLOT: cudaWarpQ1 (PNG)
########################################

set terminal pngcairo size 1200,800 enhanced font 'Verdana,12'
set output 'cudaWarpQ1.png'


plot 'cudaWarpQ1.txt' using 1:(column(2)) with points pointtype 1 \
title sprintf("p = %d", 1)


unset output

########################################
# PLOT: cudaWarpQ1 (SVG)
########################################
if (enableSVG){
     set terminal svg size 1200,800 enhanced font 'Verdana,12' 
     set output 'cudaWarpQ1.svg' 
     plot 'cudaWarpQ1.txt' using 1:(2) with points pointtype (1) \
     title sprintf("p = %d", 1)
}

unset output


########################################
# PLOT: cuda1D (PNG)
########################################

set terminal pngcairo size 1200,800 enhanced font 'Verdana,12'
set output 'cuda1D.png'

plot for [i = polyOrderBegin : polyOrderEnd] \
     'cuda1D.txt' using 1:(column(i+2 - polyOrderBegin)) \
     with points pointtype (i - polyOrderBegin + 1) \
     title sprintf("p = %d", i)

unset output

########################################
# PLOT: cuda1D (SVG)
########################################
if (enableSVG){
     set terminal svg size 1200,800 enhanced font 'Verdana,12' 
     set output 'cuda1D.svg' 
     plot for [i = polyOrderBegin : polyOrderEnd] \
     'cuda1D.txt' using 1:(column(i+2 - polyOrderBegin)) \
     with points pointtype (i - polyOrderBegin + 1) \
     title sprintf("p = %d", i)
}

unset output

########################################
# PLOT: cuda3D (PNG)
########################################

set terminal pngcairo size 1200,800 enhanced font 'Verdana,12'
set output 'cuda3D.png'

plot for [i = polyOrderBegin : polyOrderEnd] \
     'cuda3D.txt' using 1:(column(i+2 - polyOrderBegin)) \
     with points pointtype (i - polyOrderBegin + 1) \
     title sprintf("p = %d", i)

unset output

########################################
# PLOT: cuda3D (SVG)
########################################
if (enableSVG){
     set terminal svg size 1200,800 enhanced font 'Verdana,12' 
     set output 'cuda3D.svg' 
     plot for [i = polyOrderBegin : polyOrderEnd] \
     'cuda3D.txt' using 1:(column(i+2 - polyOrderBegin)) \
     with points pointtype (i - polyOrderBegin + 1) \
     title sprintf("p = %d", i)
}

unset output


########################################
# PLOT: cuda3DS (PNG)
########################################

set terminal pngcairo size 1200,800 enhanced font 'Verdana,12'
set output 'cuda3DS.png'

plot for [i = polyOrderBegin : polyOrderEnd] \
     'cuda3DS.txt' using 1:(column(i+2 - polyOrderBegin)) \
     with points pointtype (i - polyOrderBegin + 1) \
     title sprintf("p = %d", i)

unset output


########################################
# PLOT: cuda3DS (SVG)
########################################
if (enableSVG){
     set terminal svg size 1200,800 enhanced font 'Verdana,12' 
     set output 'cuda3DS.svg' 
     plot for [i = polyOrderBegin : polyOrderEnd] \
     'cuda3DS.txt' using 1:(column(i+2 - polyOrderBegin)) \
     with points pointtype (i - polyOrderBegin + 1) \
     title sprintf("p = %d", i)
}

unset output





