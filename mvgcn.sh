#!/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -l walltime=4:00:00
#PBS -N session1_default
#PBS -A course
#PBS -q ShortQ

#cd $PBS_O_WORKDIR
#Graph Convolutional Networks: one layer or two layers
declare -a method=("gcn", "2gcn")
declare -a view="dti_fact dti_rk2 dti_sl dti_tl qbi_rk2 hough"

# predefined parameters: 'method' 'is_random' 'data_type' 'kfold' 'K' 'M' 'n_epoch' 'batch_size'
python3 train_pd.py "gcn" 'False' $view 'eu' 'max' 'True' 30 128 20 32


# now loop through the above array
# for i in "${view[@]}"
# do
#    echo "$i"
#    # or do whatever with individual element of the array
#    python3 train_pd.py "gcn" 'False' $i 'True' 30 128 20 32
# done

# for m in "${M[@]}"
# do
#    echo "$m"
#    # or do whatever with individual element of the array
#    python3 train_pd.py "dti_fact" 'True' 20 $m 20 32
# done

# for k in "${K[@]}"
# do
#    echo For the support K:
#    echo "$k"
#    for m in "${M[@]}"
#    do
#        echo For the fully connected dim M:
#        echo "$m"
#        # or do whatever with individual element of the array
#        python3 train_hy.py "dti_fact" 'True' $k $m
#    done
# done


# for e in "${n_epoch[@]}"
# do
#    echo For the number of epoches:
#    echo "$e"
#    for b in "${batch_size[@]}"
#    do
#        echo For the batch size:
#        echo "$b"
#        # or do whatever with individual element of the array
#        python3 train_pd.py "dti_fact" 'True' 20 512 $e $b
#    done
# done

echo All done
