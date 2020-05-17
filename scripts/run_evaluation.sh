#!/bin/bash

echo "Reminder: add --remove_nw option in extra_args when testing WordRM models"

current_seed=0
model_path=$1

if [[ -n "$2" ]]; then
    current_seed=$2
else
    current_seed=0
fi


if [[ -n "$3" ]]; then
    max_seed=$3
else
    max_seed=10
fi

model_dir="runs"
if [[ -n "$4" ]]; then
  model_dir=$4
fi

if [[ -n "$5" ]]; then
    extra_args=$5
else
    extra_args=""
fi



while (( ${current_seed}<${max_seed} ))
do
    python run_model.py --do_eval --do_lower_case --data_dir ./data/majority_gab_dataset_25k/ --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 5 --output_dir runs/${model_path}_seed_${current_seed} --seed ${current_seed} --task_name gab --test ${extra_args}
    python run_model.py --do_eval --do_lower_case --data_dir ./data/white_supremacy/ --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 5 --output_dir ${model_dir}/${model_path}_seed_${current_seed} --seed ${current_seed} --task_name ws --test ${extra_args}
    python run_model.py --do_eval --do_lower_case --data_dir ./data/nyt_keyword_sample/ --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 5 --output_dir runs/${model_path}_seed_${current_seed} --seed ${current_seed} --task_name nyt --test ${extra_args}
    let current_seed++
done
