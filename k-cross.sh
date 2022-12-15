#!/bin/sh
#$ -cwd
#$ -N NN_2D

echo -e "started on $(date)\n"

python "./k_cross2.py" > log.txt

echo -e "\nfinished on $(date)\n"

