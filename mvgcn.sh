#!/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -l walltime=4:00:00
#PBS -N session1_default
#PBS -A course
#PBS -q ShortQ

#cd $PBS_O_WORKDIR
#Graph Convolutional Networks: one layer or two layers
declare -a view="dti_fact dti_rk2 dti_sl dti_tl qbi_rk2 hough"

# predefined parameters: 'method' 'data_type' 'is_kfold' 'K' 'M' 'n_epoch' 'batch_size'
python3 train.py "gcn" $view 'True' 30 128 20 32
echo All done
