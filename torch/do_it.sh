#!/bin/bash
#$ -cwd
#$ -N NN_2D

echo -e "started on $(date)\n"

python torch/MLP.py input_file.txt

echo -e "\nfinished on $(date)\n"
