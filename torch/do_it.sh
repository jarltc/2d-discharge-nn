#!/bin/bash
#$ -cwd
#$ -N NN_2D

echo -e "started on $(date)\n"

python ./create_model.py input_file.txt

echo -e "\nfinished on $(date)\n"

