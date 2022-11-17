#!/bin/bash
#$ -cwd
#$ -N NN_2D

echo -e "started on $(date)\n"

python "./create_model.py" >  log.txt

echo -e "\nfinished on $(date)\n"

