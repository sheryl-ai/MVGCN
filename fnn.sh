#!/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -l walltime=4:00:00
#PBS -N session1_default
#PBS -A course
#PBS -q ShortQ

#cd $PBS_O_WORKDIR
#feedforward neural networks: one layer fnn and two layers fnn
declare -a method=("fnn", "2fnn")
declare -a view=("qbi_rk2")
# declare -a view=("dti_fact" "dti_rk2" "dti_sl" "dti_tl" "qbi_fact" "qbi_rk2")

# predefined parameters: 'method' 'data_type' 'kfold' 'K' 'M' 'n_epoch' 'batch_size'
python3 train_pd.py "2fnn" $view "max" "True" 30 1024 20 32

echo done
