#!/bin/bash

# training while removing neutral words

max_seeds=10
current_seed=0

while(( $current_seed < $max_seeds ))
do
    python run_model.py --do_train --do_lower_case --data_dir ./data/majority_gab_dataset_25k/ --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 20 --early_stop 5 --output_dir runs/majority_gab_es_removenw_bal_seed_$current_seed --seed $current_seed --task_name gab --remove_nw --negative_weight 0.1
    let current_seed++
done
