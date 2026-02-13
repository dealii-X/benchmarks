#!/usr/bin/gnuplot
set terminal pdfcairo size 5in,3.5in font "sans,10"
set output ARG1 . ".pdf"

if (ARGC != 1) { print "Usage: gnuplot -c plot.gp <file>"; exit }

set logscale x
set logscale y

set format x "2^{%L}"
set grid
set key left top

set xlabel "Message Size (KB)"
set ylabel "time (s)"
set title  ARG1

plot ARG1 using 1:2 with linespoints title "Min" pt 6 ps 0.4 lc rgb "#888888", \
     ARG1 using 1:3 with linespoints title "Max" pt 6 ps 0.4 lc rgb "#888888", \
     ARG1 using 1:4 with linespoints title "Avg" pt 7 ps 0.6 lc rgb "#ff7f0e" lw 1.5
