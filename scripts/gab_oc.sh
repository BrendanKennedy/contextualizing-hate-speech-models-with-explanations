#!/bin/bash

# training with regularizing OC explanations

max_seeds=10
current_seed=0

while(( $current_seed < $max_seeds ))
do
    python run_model.py --do_train --do_lower_case --data_dir ./data/majority_gab_dataset_25k/ --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 20 --early_stop 5 --output_dir runs/majority_gab_es_reg_nb0_h1_bal_seed_$current_seed --seed $current_seed --task_name gab --reg_explanations --nb_range 0 --sample_n 1 --negative_weight 0.1 --reg_strength 0.1
    let current_seed++
done
