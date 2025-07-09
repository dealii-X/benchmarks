#!/usr/bin/env bash
# normalize.sh - Normalize kernel performance data from given file

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 input_filename"
    exit 1
fi

input_file="$1"

awk '
/Warp-Centric/         { print "Warp-Centric",        $5, $8 }
/1D_Block/             { print "1D_Block",            $5, $8 }
/3D_Block Simple Map/  { print "3D_Block_Simple_Map", $7, $10 }
/3D_Block/ && !/Simple/ { print "3D_Block",           $5, $8 }
' "$input_file"
